# Robot-host client for the V-JEPA 2-AC baseline: the plain goal-image planner, or the paper's pick-and-place protocol.
#
#   0. on a GPU node:  python -m app.vjepa_hanoi.goal_planner --fname <yaml> --checkpoint <best.pt> --host 0.0.0.0 \
#                          --goal_archive <training split .npz> [--protocol paper]
#      then tunnel it to the robot host: ssh -L 8766:<gpu node>:8766 <cluster>
#   1. optional, capture a real goal photo: arrange the goal board, then
#        EXECUTE=1 PYTHONPATH=<tower_hanoi>:<vjepa2>:$PYTHONPATH python -m app.vjepa_hanoi.arm_goal_client --execute \
#            --save-goal-image goal.npy
#   2. run a test from the current board:
#        EXECUTE=1 PYTHONPATH=<tower_hanoi>:<vjepa2>:$PYTHONPATH python -m app.vjepa_hanoi.arm_goal_client --execute \
#            --start AAAA --goal-board BAAA [--goal-image goal.npy] --log run_AAAA_BAAA.npz
#   The real arm moves only with --execute plus EXECUTE=1 (a dry run takes neither). Keep $PYTHONPATH: it carries ROS's
#   rclpy for the camera.
#
# The goal is set once (an uploaded photo, or a training frame of the goal board on the server). Every control step
# sends the current 224 px frame + measured pose + gripper coordinate, executes the first planned action (one model
# step of 0.267 s of recorded motion, moved as robot_io describes; the server only clamps it to the workspace and a
# speed limit), and repeats until the step budget is used:
# --steps, or by default 2 x the optimal number of game moves x --steps-per-move (the composition test's budget of
# twice the optimal move count). There is no success detector and no subgoal or route logic; check the board at the
# end (e.g. robot.check_board). Every step is logged to --log for analysis.
#
# --protocol paper (server started with --protocol paper) runs the V-JEPA 2 paper's pick-and-place protocol for ONE
# game move (--start and --goal-board one legal ring transfer apart). There are three goal images: the ring grasped,
# the ring above the target peg, and the ring placed. They are the server's training frames of that transition, or
# your own photos via --goal-images grasped.npy,near.npy,final.npy. The goals switch on the paper's fixed time
# schedule: --schedule, by default the training split's median expert steps per goal as reported by the server.
# The budget is the schedule total.
#   EXECUTE=1 PYTHONPATH=<tower_hanoi>:<vjepa2>:$PYTHONPATH python -m app.vjepa_hanoi.arm_goal_client --execute \
#       --protocol paper --start AAAA --goal-board BAAA --log run_paper_AAAA_BAAA.npz

import argparse
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from collections import deque

import numpy as np

from app.vjepa_hanoi.robot_io import (SPEED_CAP_M_S, START_POSE, STEP_S, STEP_SLOW_S, Robot, step_goal_time,
                                      workspace_low_floor)  # fmt: skip

EXECUTE_HELP = "real arm only: required, together with the environment variable EXECUTE=1 (run by the operator)"


def require_execute(p, args):
    """The real arm moves only with --execute plus EXECUTE=1 (tower_hanoi's hardware rule); a dry run takes neither."""
    if args.dry_run:
        if args.execute:
            p.error("--execute is for the real arm; drop it for a dry run")
    elif not (args.execute and os.environ.get("EXECUTE") == "1"):
        p.error("the real arm moves only with --execute and EXECUTE=1 in the environment; use --dry-run to rehearse")


def board_distance(start, goal):
    """Optimal number of game moves start -> goal (only used to size the step budget)."""

    def moves(b):
        for ring, peg in enumerate(b):
            if any(b[r] == peg for r in range(ring)):
                continue
            for t in "ABC":
                if t != peg and not any(b[r] == t for r in range(ring)):
                    yield b[:ring] + t + b[ring + 1 :]

    dist, queue = {start: 0}, deque([start])
    while queue:
        b = queue.popleft()
        for n in moves(b):
            if n not in dist:
                dist[n] = dist[b] + 1
                queue.append(n)
    return dist[goal]


def post(server, path, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    req = urllib.request.Request(server + path, data=buf.getvalue(), method="POST")
    try:
        # PLANNER_TIMEOUT_S: seconds to wait for one plan (default 120; H=3 at 800 samples on a 16 GB GPU needs more)
        with urllib.request.urlopen(req, timeout=float(os.environ.get("PLANNER_TIMEOUT_S", 120))) as resp:
            out = json.loads(resp.read())
    except urllib.error.HTTPError as e:  # the server answers 400 with {"error": reason}
        try:
            out = json.loads(e.read())
        except Exception:
            raise RuntimeError(f"planner: HTTP {e.code} {e.reason}") from e
    if "error" in out:
        raise RuntimeError(f"planner: {out['error']}")
    return out


def load_image(path):
    if path.endswith(".npy"):
        return np.load(path)
    import cv2

    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def paper_goal(args, p):
    """Paper protocol: set the three goals on the server; return (reply, budget, steps after which goals switch)."""
    opts = {k: True for k in ("fix_x", "hold_grasp") if getattr(args, k)}  # flags add to the server's defaults
    if args.goal_images:
        images = np.stack([load_image(f).astype(np.uint8) for f in args.goal_images.split(",")])
        goal = post(args.server, "/goal", goal_images=images, **opts)
    else:
        if not (args.start and args.goal_board) or board_distance(args.start, args.goal_board) != 1:
            p.error("--protocol paper is one game move: give --start and --goal-board one ring transfer apart")
        goal = post(args.server, "/goal", transition=f"{args.start}>{args.goal_board}", **opts)
    if goal.get("protocol") != "paper":
        p.error("the server is not running --protocol paper")
    schedule = [int(s) for s in args.schedule.split(",")] if args.schedule else goal.get("schedule")
    if not schedule or len(schedule) != goal["n_goals"]:
        p.error(f"give --schedule with one step count per goal ({goal['n_goals']})")
    switch = list(np.cumsum(schedule)[:-1])
    budget = args.steps if args.steps is not None else int(sum(schedule))
    print(f"paper protocol: {goal['n_goals']} goals, schedule {schedule} steps, options {goal.get('options')}, "
          f"planner {goal['planner']}", flush=True)  # fmt: skip
    return goal, budget, switch


LOG_KEYS = ("pose", "jaw", "action", "executed_delta", "goal_energy", "plan_energy", "scaled", "goal_index",
            "gripper_held", "gripper_intent", "plan_seconds", "move_s", "move_class", "grasp_reading", "image")


def new_log():
    return {k: [] for k in LOG_KEYS}


def save_log(path, log, **meta):
    """Every step (with its 224 px camera frame) plus run metadata, compressed."""
    arrays = {k: np.asarray(v) for k, v in log.items() if v}
    arrays.update({f"meta_{k}": np.asarray(json.dumps(v)) for k, v in meta.items()})  # JSON text: loads without pickle
    np.savez_compressed(path, **arrays)


def run_steps(robot, server, goal, budget, switch, log, out=print, seed=0, intent=None, step_log=None):
    """The control loop: observe, plan toward the active goal, execute the first action; `log` fills as it goes.

    `switch` lists the steps after which the paper protocol moves to the next goal (empty: the single goal). A
    hardware or planner error propagates; the steps already taken stay in `log`. Step k plans with CEM seed
    seed * 1000 + k (the offline check also seeds every row differently). `intent` is the gripper state the caller
    left the arm in ("open" after parking), so only real changes count as jaw changes.

    Execution follows robot_io: the low-lateral check before any of the step's motion, the gripper only on a change of
    intent (a close is checked for a missed grasp), then the move to the planned target with step_goal_time's goal
    time and the readback settle window. `step_log` (a path) gets one JSON line per planned step as it happens, so a
    killed process still leaves the run's record.
    """
    last_gripper = None  # the step of the last gripper command
    for step in range(1, budget + 1):
        gi = sum(step > s for s in switch)  # paper protocol: goals switch at fixed steps; plain: the one goal
        if gi and step - 1 == switch[gi - 1]:
            out(f"step {step}: switching to goal {gi + 1} of {goal['n_goals']}")
        image, pose, jaw = robot.observe()
        reply = post(server, "/plan", image=image.astype(np.uint8), pose=np.asarray(pose, float), jaw=float(jaw),
                     goal_index=gi if switch else -1, seed=seed * 1000 + step)  # fmt: skip
        target, new_intent = reply["target_pose"], reply["gripper_intent"]
        jaw_change = new_intent != intent
        loaded_low = new_intent == "close" and min(float(pose[2]), float(target[2])) < robot.low_floor
        move_s, move_class = step_goal_time(
            math.dist([float(v) for v in pose[:3]], [float(v) for v in target[:3]]), jaw_change,
            None if last_gripper is None else step - last_gripper - 1, loaded_low,
        )  # fmt: skip
        for k, v in (("pose", pose), ("jaw", jaw), ("action", reply["action"]), ("scaled", reply["safety_scaled"]),
                     ("executed_delta", reply["executed_delta"]), ("goal_energy", reply["goal_energy"]),
                     ("plan_energy", reply["plan_energy"]), ("goal_index", reply["goal_index"]),
                     ("gripper_held", reply.get("gripper_held", False)), ("gripper_intent", new_intent),
                     ("plan_seconds", reply["seconds"]), ("move_s", move_s), ("move_class", move_class),
                     ("grasp_reading", float("nan")), ("image", image.astype(np.uint8))):  # fmt: skip
            log[k].append(v)
        if step_log:
            row = {"step": step, "time": time.time(), "pose": [float(v) for v in pose], "jaw": float(jaw),
                   "target_pose": target, "move_s": move_s, "move_class": move_class, "seed": seed * 1000 + step}
            keep = ("action", "executed_delta", "safety_scaled", "goal_index", "gripper_intent", "gripper_held",
                    "goal_energy", "plan_energy", "seconds")
            row.update({k: reply.get(k) for k in keep})
            with open(step_log, "a") as f:
                f.write(json.dumps(row) + "\n")
        robot.check_step(pose, target)  # abort-only, before any of this step's motion
        if jaw_change:
            reading = robot.gripper(new_intent)  # a close below the low floor is checked for a missed grasp
            intent, last_gripper = new_intent, step
            if reading is not None:
                log["grasp_reading"][-1] = float(reading)
        robot.move(target, move_s, settle=True)
        if step % 10 == 0 or step == 1:
            d = 1e3 * np.asarray(reply["executed_delta"])
            out(f"step {step}/{budget}: delta [{d[0]:+.1f} {d[1]:+.1f} {d[2]:+.1f}] mm, gripper {intent}, goal "
                f"{reply['goal_index'] + 1}, energy {reply['goal_energy']:.4f}, plan {reply['seconds']:.1f} s, move "
                f"{move_s:.2f} s ({move_class}){' (speed-limited)' if reply['safety_scaled'] else ''}")
    return log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://127.0.0.1:8766")
    p.add_argument("--protocol", choices=("plain", "paper"), default="plain", help="must match the server's")
    p.add_argument("--start", help="current board, e.g. AAAA (sizes the step budget)")
    p.add_argument("--goal-board", help="goal board, e.g. CCCC (server uses its training frames as the goal image)")
    p.add_argument("--goal-image", help="goal photo (.npy 224x224x3 RGB from --save-goal-image, or .png)")
    p.add_argument("--goal-images", help="paper protocol: grasped,near-target,final photos (comma-separated)")
    p.add_argument("--schedule", help="paper protocol: steps per goal, e.g. 25,23,12 (default: the server's)")
    p.add_argument("--fix-x", dest="fix_x", action="store_true", help="deviation: hold x at 0 for this run")
    p.add_argument("--hold-grasp", dest="hold_grasp", action="store_true",
                   help="deviation (paper protocol): gripper held closed while goal 2 is active")  # fmt: skip
    p.add_argument("--save-goal-image", help="capture the current transformed frame to this .npy and exit")
    p.add_argument("--steps", type=int, default=None, help="step budget (overrides the default)")
    p.add_argument("--steps-per-move", type=int, default=120,
                   help="~32 s per game move in fps-4 steps of 0.267 s (scaled to the server model's step)")
    p.add_argument("--log", default="vjepa_plain_run.npz")
    p.add_argument("--no-park", dest="no_park", action="store_true",
                   help="start from the current pose instead of opening the gripper and parking at the start pose")
    p.add_argument("--tower-hanoi", default=None, help="tower_hanoi checkout (for its robot/config)")
    p.add_argument("--address", default="192.168.1.3")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--replay", default=None, help="dry run: HDF5 recording replayed as the camera")
    p.add_argument("--replay-episode", type=int, default=0)
    p.add_argument("--execute", action="store_true", help=EXECUTE_HELP)
    args = p.parse_args()
    require_execute(p, args)
    if args.tower_hanoi is None:
        args.tower_hanoi = os.environ.get("TOWER_HANOI", os.path.expanduser("~/ws/tower_hanoi"))
    if args.dry_run and not args.replay:
        p.error("--dry-run needs --replay <recording.h5>")
    if not args.save_goal_image:  # the move-time rule (robot_io.step_goal_time) holds 36 mm/s only up to 36 mm steps
        with urllib.request.urlopen(args.server + "/health", timeout=30) as resp:
            health = json.loads(resp.read())
        if float(health.get("max_step_m", 0.0)) > SPEED_CAP_M_S * STEP_SLOW_S + 1e-9:
            p.error(f"the server allows {health['max_step_m'] * 1e3:.1f} mm steps (max_speed x a model step of "
                    f"{health.get('step_s', STEP_S):.3f} s); robot_io moves a step in at most {STEP_SLOW_S:.1f} s, so "
                    f"only steps up to {SPEED_CAP_M_S * STEP_SLOW_S * 1e3:.0f} mm stay under the proven 36 mm/s")
        server_floor = (health.get("planner") or {}).get("low_floor")  # goal_planner --guard_floor_mm
        if server_floor is not None:  # the planner must avoid this arm's low-lateral guard, not another arm's
            floor = workspace_low_floor(args.tower_hanoi)
            if abs(float(server_floor) - floor) > 1e-3:
                p.error(f"the server plans with a low floor of {float(server_floor) * 1e3:.2f} mm "
                        f"(--guard_floor_mm) but this arm's is {floor * 1e3:.2f} mm (robot_io: transit_z - 25 mm); "
                        f"restart it with --guard_floor_mm {floor * 1e3:.2f}")  # fmt: skip

    robot = Robot(args)
    try:
        if args.save_goal_image:
            image, _, _ = robot.observe()
            np.save(args.save_goal_image, image)
            print(f"saved goal image {image.shape} -> {args.save_goal_image}")
            return 0
        if args.protocol == "paper":
            goal, budget, switch = paper_goal(args, p)
        else:
            if not (args.goal_board or args.goal_image):
                p.error("give --goal-board and/or --goal-image")
            if args.hold_grasp:
                p.error("--hold-grasp is a paper-protocol option")
            opts = {"fix_x": True} if args.fix_x else {}
            goal = (
                post(args.server, "/goal", goal_image=load_image(args.goal_image).astype(np.uint8), **opts)
                if args.goal_image
                else post(args.server, "/goal", goal_board=args.goal_board, **opts)
            )
            budget, switch = args.steps, []
            if budget is None:
                if not (args.start and args.goal_board):
                    p.error("the default budget needs --start and --goal-board (or pass --steps)")
                per_move = math.ceil(args.steps_per_move * 8 / int(goal.get("step_rows", 8)))  # in the model's steps
                budget = 2 * board_distance(args.start, args.goal_board) * per_move
        if args.dry_run:
            robot.replay.step_rows = int(goal.get("step_rows", 8))  # one model step of the server's model per frame
        print(f"goal: {goal['goal']}; budget {budget} model steps ({budget * goal.get('step_s', STEP_S):.0f} s of "
              f"recorded time)", flush=True)  # fmt: skip
        if not args.no_park:  # the server refuses poses outside the trained workspace, so start where training did
            robot.gripper("open")
            robot.park(START_POSE)
        log = new_log()
        run_steps(robot, args.server, goal, budget, switch, log, intent=None if args.no_park else "open")
        print("budget used; check the board (e.g. robot.check_board) and record success/failure.", flush=True)
        return 0
    finally:
        try:
            if "log" in locals() and log["pose"]:
                save_log(args.log, log)
                print(f"step log -> {args.log}")
        finally:
            robot.close()  # whatever the log save does


if __name__ == "__main__":
    sys.exit(main())

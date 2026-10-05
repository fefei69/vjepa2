# Robot-host client for the PLAIN V-JEPA 2-AC baseline: one goal image, the repo planner, nothing else.
#
#   0. on a GPU node:  python -m app.vjepa_hanoi.goal_planner --fname <yaml> --checkpoint <best.pt> --host 0.0.0.0 \
#                          --goal_archive <openpi play_train.npz>
#      then tunnel it to the robot host: ssh -L 8766:<gpu node>:8766 <cluster>
#   1. optional, capture a real goal photo: arrange the goal board, then
#        PYTHONPATH=<tower_hanoi>:<vjepa2> python -m app.vjepa_hanoi.arm_goal_client --save-goal-image goal.npy
#   2. run a test from the current board:
#        PYTHONPATH=<tower_hanoi>:<vjepa2> python -m app.vjepa_hanoi.arm_goal_client --start AAAA \
#            --goal-board BAAA [--goal-image goal.npy] --log run_AAAA_BAAA.npz
#
# The goal is set once (an uploaded photo, or the goal board's training frames on the server). Every control step
# sends the current 224 px frame + measured pose + gripper coordinate, executes the first planned action (one 0.267 s
# step; the server only clamps it to the workspace and a speed limit), and repeats until the step budget is used:
# --steps, or by default 2 x the optimal number of game moves x --steps-per-move (the composition test's budget of
# twice the optimal move count). There is no success detector and no subgoal or route logic; check the board at the
# end (e.g. robot.check_board). Every step is logged to --log for analysis.

import argparse
import io
import json
import sys
import urllib.request
from collections import deque

import numpy as np

from app.vjepa_hanoi.robot_io import STEP_S, Robot


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
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read())
    if "error" in out:
        raise RuntimeError(f"planner: {out['error']}")
    return out


def load_image(path):
    if path.endswith(".npy"):
        return np.load(path)
    import cv2

    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://127.0.0.1:8766")
    p.add_argument("--start", help="current board, e.g. AAAA (sizes the step budget)")
    p.add_argument("--goal-board", help="goal board, e.g. CCCC (server uses its training frames as the goal image)")
    p.add_argument("--goal-image", help="goal photo (.npy 224x224x3 RGB from --save-goal-image, or .png)")
    p.add_argument("--save-goal-image", help="capture the current transformed frame to this .npy and exit")
    p.add_argument("--steps", type=int, default=None, help="step budget (overrides the default)")
    p.add_argument("--steps-per-move", type=int, default=120, help="~32 s per game move at 0.267 s per step")
    p.add_argument("--log", default="vjepa_plain_run.npz")
    p.add_argument("--tower-hanoi", default=None, help="tower_hanoi checkout (for its robot/config)")
    p.add_argument("--address", default="192.168.1.3")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--replay", default=None, help="dry run: HDF5 recording replayed as the camera")
    p.add_argument("--replay-episode", type=int, default=0)
    args = p.parse_args()
    if args.tower_hanoi is None:
        import os

        args.tower_hanoi = os.environ.get("TOWER_HANOI", os.path.expanduser("~/ws/tower_hanoi"))
    if args.dry_run and not args.replay:
        p.error("--dry-run needs --replay <recording.h5>")

    robot = Robot(args)
    try:
        if args.save_goal_image:
            image, _, _ = robot.observe()
            np.save(args.save_goal_image, image)
            print(f"saved goal image {image.shape} -> {args.save_goal_image}")
            return 0
        if not (args.goal_board or args.goal_image):
            p.error("give --goal-board and/or --goal-image")
        goal = (
            post(args.server, "/goal", goal_image=load_image(args.goal_image).astype(np.uint8))
            if args.goal_image
            else post(args.server, "/goal", goal_board=args.goal_board)
        )
        budget = args.steps
        if budget is None:
            if not (args.start and args.goal_board):
                p.error("the default budget needs --start and --goal-board (or pass --steps)")
            budget = 2 * board_distance(args.start, args.goal_board) * args.steps_per_move
        print(f"goal: {goal['goal']}; budget {budget} steps ({budget * STEP_S:.0f} s)", flush=True)
        log = {k: [] for k in ("pose", "jaw", "action", "executed_delta", "goal_energy", "plan_energy", "scaled")}
        intent = None
        for step in range(1, budget + 1):
            image, pose, jaw = robot.observe()
            out = post(
                args.server, "/plan", image=image.astype(np.uint8), pose=np.asarray(pose, float), jaw=float(jaw)
            )
            if out["gripper_intent"] != intent:
                robot.gripper(out["gripper_intent"])
                intent = out["gripper_intent"]
            robot.move(out["target_pose"], STEP_S)
            for k, v in (("pose", pose), ("jaw", jaw), ("action", out["action"]), ("scaled", out["safety_scaled"]),
                         ("executed_delta", out["executed_delta"]), ("goal_energy", out["goal_energy"]),
                         ("plan_energy", out["plan_energy"])):  # fmt: skip
                log[k].append(v)
            if step % 10 == 0 or step == 1:
                d = 1e3 * np.asarray(out["executed_delta"])
                print(f"step {step}/{budget}: delta [{d[0]:+.1f} {d[1]:+.1f} {d[2]:+.1f}] mm, gripper {intent}, "
                      f"goal energy {out['goal_energy']:.4f}{' (speed-limited)' if out['safety_scaled'] else ''}",
                      flush=True)  # fmt: skip
        print("budget used; check the board (e.g. robot.check_board) and record success/failure.", flush=True)
        return 0
    finally:
        if "log" in locals() and log["pose"]:
            np.savez_compressed(args.log, **{k: np.asarray(v) for k, v in log.items()})
            print(f"step log -> {args.log}")
        robot.close()


if __name__ == "__main__":
    sys.exit(main())

# Robot-host client for the ORACLE subgoal planner (symbolic route + demo subgoals; see README.md -- not the baseline).
#
#   on the robot host (tower_hanoi environment, planner reachable e.g. through `ssh -L 8765:<gpu node>:8765`):
#     PYTHONPATH=<tower_hanoi>:<vjepa2> python -m app.vjepa_hanoi_oracle.arm_client --start AAAA --goal CCCC
#   rehearsal without hardware (DryRunArm, a recorded walk replayed as the camera, no ROS):
#     PYTHONPATH=<tower_hanoi>:<vjepa2> python -m app.vjepa_hanoi_oracle.arm_client --start AAAA --goal BAAA \
#         --dry-run \
#         --replay /scratch/cw5167/datasets/hanoi_wm_20260924_210743.h5
#
# Every ring transfer is seven subgoals, one per motion stage (app/vjepa_hanoi_oracle/subgoals.py). Each control
# step sends the current 224 px frame (the recording's square_roi transform), measured pose and gripper coordinate,
# receives one planned 0.267 s step, and executes it as a blocking Cartesian move (x held, workspace-clamped by
# the server) plus a gripper command when the intent changes. A subgoal is done when the planned y-z step stays
# below --done-mm for --done-steps consecutive steps with the phase's gripper intent; if --max-steps pass first,
# the run STOPS for the operator instead of continuing on a guess.
#
# Only stdlib + numpy here; the tower_hanoi modules (robot.arm, robot.camera, robot.wm_dataset) are imported from
# the robot repository so the hardware path is exactly the one the data collector uses.

import argparse
import io
import json
import os
import sys
import urllib.request
from collections import deque

import numpy as np

from app.vjepa_hanoi.robot_io import STEP_S, Robot

PHASE_INTENT = {"descended": "open", "closed": "close", "lifted": "close", "above_target": "close",
                "inserted": "close", "released": "open", "retreated": "open"}  # fmt: skip


def legal_moves(board):
    """board: peg letter per ring, smallest first. Yields boards after one legal move."""
    for ring, peg in enumerate(board):
        if any(board[r] == peg for r in range(ring)):  # a smaller ring sits on top
            continue
        for target in "ABC":
            if target != peg and not any(board[r] == target for r in range(ring)):
                yield board[:ring] + target + board[ring + 1 :]


def route(start, goal):
    """Shortest legal board sequence start -> goal (breadth-first over the 81 boards)."""
    prev, queue = {start: None}, deque([start])
    while queue:
        b = queue.popleft()
        if b == goal:
            break
        for n in legal_moves(b):
            if n not in prev:
                prev[n] = b
                queue.append(n)
    path, b = [], goal
    while b is not None:
        path.append(b)
        b = prev[b]
    return path[::-1]


def request_plan(server, image, pose, jaw, subgoal):
    buf = io.BytesIO()
    np.savez(buf, image=image.astype(np.uint8), pose=np.asarray(pose, float), jaw=float(jaw), subgoal=subgoal)
    req = urllib.request.Request(server + "/plan", data=buf.getvalue(), method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.loads(resp.read())
    if "error" in out:
        raise RuntimeError(f"planner: {out['error']}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://127.0.0.1:8765")
    p.add_argument("--start", required=True, help="current board, e.g. AAAA (peg per ring, smallest first)")
    p.add_argument("--goal", required=True, help="goal board, e.g. CCCC")
    p.add_argument("--max-steps", type=int, default=60, help="per subgoal; exceeding it stops the run")
    p.add_argument("--done-mm", type=float, default=1.5)
    p.add_argument("--done-steps", type=int, default=2)
    p.add_argument("--step-s", type=float, default=STEP_S, help="duration of each executed step")
    p.add_argument("--tower-hanoi", default=os.environ.get("TOWER_HANOI", os.path.expanduser("~/ws/tower_hanoi")))
    p.add_argument("--address", default="192.168.1.3")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--replay", default=None, help="dry run: HDF5 recording whose frames stand in for the camera")
    p.add_argument("--replay-episode", type=int, default=0)
    args = p.parse_args()
    if args.dry_run and not args.replay:
        p.error("--dry-run needs --replay <recording.h5>")

    boards = route(args.start, args.goal)
    plan = [(f"{a}>{b}", ph) for a, b in zip(boards, boards[1:]) for ph in PHASE_INTENT]
    print(f"route {' -> '.join(boards)} ({len(boards) - 1} moves, {len(plan)} subgoals)", flush=True)
    robot = Robot(args)
    intent = None
    try:
        for key, phase in plan:
            subgoal, done = f"{key}:{phase}", 0
            for step in range(1, args.max_steps + 1):  # noqa: B007
                image, pose, jaw = robot.observe()
                out = request_plan(args.server, image, pose, jaw, subgoal)
                dyz = 1e3 * float(np.linalg.norm(np.asarray(out["action"][1:3])))
                print(f"{subgoal} step {step}: dyz {dyz:5.1f} mm, gripper {out['gripper_intent']}, "
                      f"goal energy {out['goal_energy']:.4f}, plan {out['seconds']:.2f} s", flush=True)  # fmt: skip
                if out["gripper_intent"] != intent:
                    robot.gripper(out["gripper_intent"])
                    intent = out["gripper_intent"]
                robot.move(out["target_pose"], args.step_s)
                done = done + 1 if (dyz < args.done_mm and intent == PHASE_INTENT[phase]) else 0
                if done >= args.done_steps:
                    break
            else:
                print(f"STOP: {subgoal} not reached in {args.max_steps} steps; check the board and arm.", flush=True)
                return 2
        print(f"route finished: expected board {args.goal} (verify with robot.check_board)", flush=True)
        return 0
    finally:
        robot.close()


if __name__ == "__main__":
    sys.exit(main())

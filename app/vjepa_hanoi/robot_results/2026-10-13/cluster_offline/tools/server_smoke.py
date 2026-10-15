# The arm's planning server (goal_planner.GoalPlanner, as started on the robot from a bundle) driven through
# descent_replay's closed loop on recorded frames: checks that the server code path with the chosen options gives the
# same descend + grasp + lift as the offline replay, without the robot or HTTP.
#
#   python server_smoke.py --bundle <robot_bundle_dir> --sampling cat --guard_floor_mm 166.13 --max_speed 0.10 \
#       --out <json>
#
# The goal frames and the hover schedule are the bundle's (set_goal(transition=..., subgoals=stages)), as
# one_move_test uses them; each step plans with GoalPlanner.plan on the recorded frame nearest the current pose, and the
# executed step is the server's (its box clamp and --max_speed cap: target_pose), then robot_io's low-lateral guard.
import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from descent_replay import K, replay_case  # noqa: E402

from app.vjepa_hanoi.goal_planner import GoalPlanner  # noqa: E402
from app.vjepa_hanoi.paper_protocol_eval import goal_stages, moves  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--archive", default=f"{K}/test.npz")
    p.add_argument("--sampling", default="cat")
    p.add_argument("--guard_floor_mm", type=float, default=None)
    p.add_argument("--max_speed", type=float, default=0.10)
    p.add_argument("--hold_xy", action="store_true")
    p.add_argument("--hold_grasp", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    b = args.bundle
    planner = GoalPlanner(f"{b}/config.yaml", f"{b}/weights.pt", goal_bundle=f"{b}/goals.npz", max_speed=args.max_speed,
                          protocol="paper", sampling=args.sampling, guard_floor_mm=args.guard_floor_mm,
                          hold_xy=args.hold_xy, hold_grasp=args.hold_grasp, subgoals="stages")  # fmt: skip
    gstages = goal_stages("stages", 8)
    test_moves, test_paths, _ = moves(args.archive)
    out = {"bundle": b, "args": vars(args), "max_step_m": planner.max_step, "cases": []}
    for m in test_moves:
        g = planner.set_goal(transition=m["transition"], subgoals="stages")
        reach = np.cumsum(g["hover_schedule"]).tolist()
        h = h5py.File(test_paths[m["file"]], "r")
        last = {}

        def plan_step(row, s7, gi):
            pose = np.concatenate([s7[:3], h["state"][row, 3:6]])
            r = planner.plan(h["pixels"][row], pose, float(h["proprio"][row, 6]), goal_index=gi - 1, seed=len(last))
            last[len(last)] = r
            a = np.asarray(r["action"], dtype=np.float64)
            a[:3] = np.asarray(r["executed_delta"])  # the server's executed step (box clamp + --max_speed cap)
            return a

        case = replay_case(m, h, planner.fstp, reach, gstages, plan_step, 2.0, 15.0)
        h.close()
        case["server_replies"] = [{k: v for k, v in r.items() if k in ("action", "executed_delta", "safety_scaled",
                                   "gripper_intent", "goal_index", "axis_held")} for r in last.values()]  # fmt: skip
        out["cases"].append({"transition": m["transition"], "hover_schedule": g["hover_schedule"]} | case)
        print(f"{m['transition']}: {case['outcome']} in {case['steps']} steps, close {case['close']}, "
              f"scaled steps {sum(r['safety_scaled'] for r in last.values())}", flush=True)  # fmt: skip
    out["success"] = sum(c["outcome"] == "success" for c in out["cases"])
    print(f"== success {out['success']}/{len(out['cases'])} (server max step {planner.max_step * 1e3:.1f} mm)")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

# Whole-move table from paper_protocol_eval runs over every row of the six held-out first moves (all_*.json).
#
#   PYTHONPATH=. python move_report.py <dir with all_*.json>
#
# Per motion stage (1 approach .. 8 release), on rows one model step apart, each planned from the recorded frame and
# state (open loop along the expert's path):
#   grip ok    share of rows whose gripper intent after the planned action equals the expert's after its next step
#   lat err    median |planned dxy - expert dxy| (mm)
#   dz err     median |planned dz - expert dz| (mm)
#   cos        median 3D cosine between the planned and the expert step (rows where the expert moves >= 5 mm)
import argparse
import glob
import json
import os

import numpy as np

from app.vjepa_hanoi.eval import STAGES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--prefix", default="all_")
    args = p.parse_args()
    for js in sorted(glob.glob(os.path.join(args.dir, f"{args.prefix}*.json"))):
        r = json.load(open(js))
        z = np.load(js.replace(".json", "_rows.npz"))
        pl, tr, c = z["plan"], z["true"], z["closedness"]
        intent = np.clip(c + pl[:, 6], 0, 1) > 0.5
        expert = np.clip(c + tr[:, 6], 0, 1) > 0.5
        lat = np.linalg.norm(pl[:, :2] - tr[:, :2], axis=1) * 1e3
        dz = np.abs(pl[:, 2] - tr[:, 2]) * 1e3
        tm = np.linalg.norm(tr[:, :3], axis=1)
        cos = np.sum(pl[:, :3] * tr[:, :3], 1) / (np.linalg.norm(pl[:, :3], axis=1) * tm + 1e-9)
        aids = ",".join(k for k in ("hold_open", "hold_xy", "hold_grasp", "fix_x") if r.get(k)) or "none"
        print(f"\n### {os.path.basename(js)[len(args.prefix):-5]}: goals {r['goals']}, H {r['planner']['rollout']}, "
              f"aids {aids}, schedule {r['time_schedule_steps_per_goal']}, {len(pl)} rows")  # fmt: skip
        print("| Stage | rows | grip ok | lat err (mm) | dz err (mm) | cos (moving rows) |\n|---|---|---|---|---|---|")
        for s in np.unique(z["stage"]):
            m = z["stage"] == s
            mv = m & (tm >= 0.005)
            print(f"| {s} {STAGES[int(s)]} | {m.sum()} | {np.mean(intent[m] == expert[m]):.2f} | {np.median(lat[m]):.1f} "
                  f"| {np.median(dz[m]):.1f} | {np.median(cos[mv]) if mv.any() else float('nan'):.2f} |")  # fmt: skip
        print(f"| all | {len(pl)} | {np.mean(intent == expert):.2f} | {np.median(lat):.1f} | {np.median(dz):.1f} | "
              f"{np.median(cos[tm >= 0.005]):.2f} |")


if __name__ == "__main__":
    main()

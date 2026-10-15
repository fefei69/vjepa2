# One open-loop table across models: per setting and model, the whole-move row of move_report and the descend/grasp
# columns of descent_report (same definitions, see those headers).
#
#   PYTHONPATH=. python compare_report.py <ppe dir> --models all_:fps4 fps1_all_:fps1 efps1_all_:efps1 \
#       --settings base_h1 stages_h1 cat_stages_h1
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from descent_report import TEST, row_metrics  # noqa: E402

from app.vjepa_hanoi.paper_protocol_eval import moves  # noqa: E402


def move_all(z):
    pl, tr, c = z["plan"], z["true"], z["closedness"]
    intent, expert = np.clip(c + pl[:, 6], 0, 1) > 0.5, np.clip(c + tr[:, 6], 0, 1) > 0.5
    tm = np.linalg.norm(tr[:, :3], axis=1)
    cos = np.sum(pl[:, :3] * tr[:, :3], 1) / (np.linalg.norm(pl[:, :3], axis=1) * tm + 1e-9)
    return (np.mean(intent == expert), np.median(np.linalg.norm(pl[:, :2] - tr[:, :2], axis=1)) * 1e3,
            np.median(np.abs(pl[:, 2] - tr[:, 2])) * 1e3, np.median(cos[tm >= 0.005]))  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--models", nargs="+", required=True, help="prefix:name")
    p.add_argument("--settings", nargs="+", required=True)
    args = p.parse_args()
    test_moves, test_paths, _ = moves(TEST)
    transitions = {m["file"]: m["transition"] for m in test_moves}
    print("| Setting | Model | grip ok | lat err (mm) | dz err (mm) | cos | hover close | early close | close at bottom "
          "| descent/step plan vs expert (mm) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in args.settings:
        for pm in args.models:
            prefix, name = pm.split(":")
            path = os.path.join(args.dir, f"{prefix}{s}_rows.npz")
            if not os.path.exists(path):
                continue
            z = np.load(path)
            g, lat, dz, cos = move_all(z)
            d = row_metrics(z, transitions)
            print(f"| {s} | {name} | {g:.2f} | {lat:.1f} | {dz:.1f} | {cos:.2f} | {d['hover_close']}/{d['cases']} | "
                  f"{d['early_close']:.2f} | {d['close_at_bottom']:.2f} | {d['descent_plan']:+.1f} vs "
                  f"{d['descent_expert']:+.1f} |")  # fmt: skip


if __name__ == "__main__":
    main()

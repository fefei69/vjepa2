# Descend/grasp table from paper_protocol_eval runs (their *_rows.npz), one line per setting.
#
#   python descent_report.py <dir with descent_*.json> [--test_archive <test.npz>]
#
# Rows are the held-out first moves' rows in stage 3 (descend) and 4 (grasp), one model step apart, each planned
# from the recorded frame and state (open loop along the expert's path). Columns:
#   hover close     cases (of 6) whose first descend row (1 raw row below the hover) gets gripper intent "close"
#   early close     share of descend rows more than 15 mm above the grasp height (the end of stage 3) with "close"
#   close at bottom share of grasp rows (stage 4) with "close" (the expert closes in this stage)
#   sideways        planned |dx, dy| (mm), median and p90, on descend + grasp rows within 40 mm of the grasp height
#                   (near the rods) while a descend/grasp goal is active; the expert's is ~0
#   after switch    planned |dx, dy| (mm, median) on grasp rows where the time schedule has already switched to the
#                   next goal (lift or transit) while the held-out demo is still grasping (one row per case at most)
#   descent/step    median planned dz on descend rows vs the expert's median dz on the same rows (mm per step)
# Label: B = paper baseline, P = planner deviation (horizon, gripper std, per-stage goals, a model step other than fps
# 4's 8 raw rows), A = task aid. For an fps-1 model use --prefix fps1_descent_ (rows every 8 raw rows, see offline.sbatch).
# "hover close" is the first evaluated descend row, at most --row_stride raw rows below the hover.
import argparse
import glob
import json
import os

import numpy as np

from app.vjepa_hanoi.paper_protocol_eval import moves

TEST = "/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/data/expert6_first_move/test.npz"


def label(r):
    if r["hold_open"] or r["hold_xy"] or r["hold_grasp"] or r["fix_x"]:
        return "A"
    pl = r["planner"]
    if r["goals"] != "paper3" or pl["rollout"] != 1 or pl["samples"] != 800 or "gripper_std" in pl \
            or r.get("step_rows", 8) != 8:
        return "P"
    return "B"


def row_metrics(z, transitions):
    st, zz = z["stage"], z["z"] * 1e3
    close = np.clip(z["closedness"] + z["plan"][:, 6], 0, 1) > 0.5
    lat = np.linalg.norm(z["plan"][:, :2], axis=1) * 1e3
    dz, tdz = z["plan"][:, 2] * 1e3, z["true"][:, 2] * 1e3
    out = {"hover_close": 0, "cases": 0, "per_case_hover": {}}
    grasp_z = {}
    for f in np.unique(z["file_index"]):
        sel = (z["file_index"] == f) & (st == 4)
        grasp_z[f] = float(np.median(zz[sel])) if sel.any() else float(zz[z["file_index"] == f].min())
    early, near = [], np.zeros(len(st), bool)
    for f in np.unique(z["file_index"]):
        rows = np.nonzero((z["file_index"] == f) & (st == 3))[0]
        if len(rows):
            first = rows[np.argmin(z["row"][rows])]
            out["cases"] += 1
            out["hover_close"] += int(close[first])
            out["per_case_hover"][transitions[int(f)]] = f"{'close' if close[first] else 'open'} {z['plan'][first, 6]:+.2f}"
        early += [i for i in rows if zz[i] > grasp_z[f] + 15]
        near |= (z["file_index"] == f) & (zz <= grasp_z[f] + 40)
    d3, g4 = st == 3, st == 4
    out |= {
        "early_close": float(np.mean(close[early])) if early else float("nan"),
        "close_at_bottom": float(np.mean(close[g4])) if g4.any() else float("nan"),
        "sideways_median": float(np.median(lat[near & (z["goal_stage"] <= 4)])),
        "sideways_p90": float(np.percentile(lat[near & (z["goal_stage"] <= 4)], 90)),
        "after_switch": f"{np.median(lat[z['goal_stage'] > 4]):.0f} mm ({int((z['goal_stage'] > 4).sum())} rows)"
        if (z["goal_stage"] > 4).any() else "-",
        "descent_plan": float(np.median(dz[d3])),
        "descent_expert": float(np.median(tdz[d3])),
        "rows": int(len(st)),
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--test_archive", default=TEST)
    p.add_argument("--prefix", default="descent_")
    args = p.parse_args()
    test, _, _ = moves(args.test_archive)
    transitions = {m["file"]: m["transition"] for m in test}
    lines = ["| Setting | Label | H | Goals | gripper std | Aids | hover close (of 6) | early close | close at bottom "
             "| sideways near rods, median / p90 (mm) | after switch | descent/step plan vs expert (mm) | rows |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    per_case = []
    for js in sorted(glob.glob(os.path.join(args.dir, f"{args.prefix}*.json"))):
        r = json.load(open(js))
        z = np.load(js.replace(".json", "_rows.npz"))
        m = row_metrics(z, transitions)
        name = os.path.basename(js)[len(args.prefix):-5]
        aids = ",".join(k for k in ("hold_open", "hold_xy", "hold_grasp", "fix_x") if r.get(k)) or "-"
        pl = r["planner"]
        lines.append(f"| {name} | {label(r)} | {pl['rollout']} | {r['goals']} | {pl.get('gripper_std', 1.0)} | {aids} "
                     f"| {m['hover_close']}/{m['cases']} | {m['early_close']:.2f} | {m['close_at_bottom']:.2f} "
                     f"| {m['sideways_median']:.1f} / {m['sideways_p90']:.1f} | {m['after_switch']} | {m['descent_plan']:+.1f} vs "
                     f"{m['descent_expert']:+.1f} | {m['rows']} |")  # fmt: skip
        per_case.append(f"{name}: " + ", ".join(f"{k} {v}" for k, v in sorted(m["per_case_hover"].items())))
    print("\n".join(lines))
    print("\nfirst descend row per case (gripper intent, planned gripper action):")
    print("\n".join(per_case))


if __name__ == "__main__":
    main()

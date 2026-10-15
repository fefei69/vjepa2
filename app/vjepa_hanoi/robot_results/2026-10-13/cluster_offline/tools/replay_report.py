# Table of descent_replay results: one row per setting, one column per model (successes of 6 and the failure modes).
#
#   python replay_report.py <dir with *_replay*.json> [--until release]
# --until lift (default): runs to the 30 mm lift; release: whole-move runs (descent_replay --until release).
import argparse
import glob
import json
import os
from collections import Counter

SHORT = {"success": "ok", "off path": "off", "closed early": "early", "dropped": "drop", "reopened": "reopen",
         "lateral guard": "guard", "timeout": "time"}  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--until", choices=("lift", "release"), default="lift")
    args = p.parse_args()
    table, models, order = {}, [], []
    for path in sorted(glob.glob(os.path.join(args.dir, "*_replay*.json"))):
        model = os.path.basename(path).split("_replay")[0]
        if model not in models:
            models.append(model)
        with open(path) as f:
            r = json.load(f)
        for name, s in r["settings"].items():
            if s["cases"] and s["cases"][0].get("until", "lift") != args.until:
                continue
            if name not in order:
                order.append(name)
            modes = Counter(SHORT[c["outcome"]] for c in s["cases"] if c["outcome"] != "success")
            steps = [c["steps"] for c in s["cases"] if c["outcome"] == "success"]
            cell = f"**{s['success']}/{len(s['cases'])}**" + (f" ({', '.join(f'{k} {v}' for k, v in modes.items())})"
                                                             if modes else "")  # fmt: skip
            if steps:
                cell += f", {sum(steps) / len(steps):.1f} steps"
            table[(name, model)] = cell
    models = [m for m in models if any(k[1] == m for k in table)]
    print("| Setting | " + " | ".join(models) + " |")
    print("|---|" + "---|" * len(models))
    for name in order:
        print(f"| {name} | " + " | ".join(table.get((name, m), "") for m in models) + " |")
    print("\nfailure modes: off = left the recorded path (> 15 mm from every recorded pose); guard = robot_io's "
          "low-lateral guard would abort (> 3 mm sideways with an end below 166 mm); early = closed > 15 mm above the "
          "grasp height; drop / reopen = opened after / without a grasp; time = not done within 2x (or --max_steps x) the expert's steps. "
          "steps: model steps from the hover to the 30 mm lift (whole move: to the release), successes.")


if __name__ == "__main__":
    main()

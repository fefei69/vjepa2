# Per-move breakdown of a descent_replay run over every H15 move (--archive data/expert6_single_move/test.npz):
#   python h15_replay_report.py replay/efps1_replay_h15.json [more.json ...]
# For each setting: success over all 90 held-out moves, by move number (1-15), by ring (1 = smallest), by task, and the
# six first moves alone (the H1 cases), plus the outcome counts. Moves are placed by h15_test.solve (each of the 90
# transitions belongs to exactly one task).
import collections
import json
import sys

from app.vjepa_hanoi.h15_test import TASKS, moving_ring, solve

WHERE = {mv: (t, k) for t in TASKS for k, mv in enumerate(solve(*t.split(">")), 1)}


def frac(xs):
    return f"{sum(xs)}/{len(xs)}"


for path in sys.argv[1:]:
    r = json.load(open(path))
    print(f"\n# {path}\ncheckpoint {r['checkpoint']}\ngoal archive {r.get('goal_archive')}, codebook "
          f"{r.get('codebook_archive')}, until {r.get('until')}")  # fmt: skip
    for name, s in r["settings"].items():
        cases = s["cases"]
        ok = {c["transition"]: c["outcome"] == "success" for c in cases}
        by_k, by_ring, by_task = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
        for mv, good in ok.items():
            t, k = WHERE[mv]
            by_k[k].append(good)
            by_ring[moving_ring(mv)[0]].append(good)
            by_task[t].append(good)
        first = [ok[solve(*t.split(">"))[0]] for t in TASKS if solve(*t.split(">"))[0] in ok]
        print(f"\n## {name}: {frac(list(ok.values()))} moves; first moves (H1 cases) {frac(first)}; later moves "
              f"{frac([g for k, v in by_k.items() if k > 1 for g in v])}")  # fmt: skip
        print("  by move:  " + "  ".join(f"{k}:{frac(by_k[k])}" for k in sorted(by_k)))
        print("  by ring:  " + "  ".join(f"ring {k}: {frac(by_ring[k])}" for k in sorted(by_ring)))
        print("  by task:  " + "  ".join(f"{t}: {frac(by_task[t])}" for t in TASKS if t in by_task))
        print("  outcomes: " + ", ".join(f"{o} {n}" for o, n in collections.Counter(c["outcome"] for c in cases).most_common()))

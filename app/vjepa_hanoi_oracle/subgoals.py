# Subgoal images for ORACLE planning of Hanoi moves (privileged aid; see README.md -- not the baseline).
#
#   python -m app.vjepa_hanoi_oracle.subgoals --archive <openpi play_train.npz> --out subgoals.npz
#
# The planner only looks ~0.3 s ahead and heads straight for its goal image, so every subgoal must lie in the
# direction of the current motion: a ring transfer is split into seven subgoals, one at the end of each motion stage,
# taken from the TRAINING split only (never validation/test):
#   descended (end of stage 3 descend_source)  closed (4 grasp)  lifted (5 lift)  above_target (6 transit)
#   inserted (7 insert_target)  released (8 release)  retreated (9 retreat)
# (A coarser 3-subgoal version sent the arm UP while descending and inserting: its goals showed the arm raised.)
# Exemplars are keyed by the move's transition "<board before>><board after>" (e.g. "AAAA>BAAA"; boards list the
# peg of rings 1-4, smallest first) and phase. Only (file, row) references are stored; pixels are read from the
# raw recordings when the planner starts.

import argparse
import itertools
from collections import defaultdict

import numpy as np

# phase -> motion stage whose last usable row is the subgoal, in execution order
PHASES = {"descended": 3, "closed": 4, "lifted": 5, "above_target": 6, "inserted": 7, "released": 8, "retreated": 9}
BOARDS = ["".join(b) for b in itertools.product("ABC", repeat=4)]  # board index order used by the archives
# motion stage of the current row -> the phase to plan toward (stages 10-12 return/hold are not part of a transfer)
STAGE_TO_PHASE = {1: "descended", 2: "descended", 3: "descended", 4: "closed", 5: "lifted", 6: "above_target",
                  7: "inserted", 8: "released", 9: "retreated"}  # fmt: skip


def move_table(archive_path):
    """One entry per (segment, move): transition key, rows per motion stage (usable observation rows only)."""
    z = np.load(archive_path)
    obs, fi, seg = z["source_observation_indices"], z["file_indices"], z["segment_bounds"]
    board, move, stage = z["board_indices"], z["move_indices"], z["motion_stages"]
    groups = defaultdict(list)
    for i in range(len(obs)):
        groups[(int(fi[i]), int(seg[i, 0]), int(move[i]))].append(i)
    moves = []
    for (f, lo, m), idx in groups.items():
        idx = np.asarray(sorted(idx, key=lambda j: obs[j]))
        before, after = BOARDS[board[idx[0]]], BOARDS[board[idx[-1]]]
        if before == after:  # terminal hold / padding rows
            continue
        moves.append(
            {
                "key": f"{before}>{after}",
                "file": f,
                "rows": obs[idx],
                "stages": stage[idx],
                "segment": (lo, int(seg[idx[0], 1])),
            }
        )
    return moves, [str(p) for p in z["source_paths"]]


def build(archive_path):
    moves, paths = move_table(archive_path)
    refs = defaultdict(list)
    for mv in moves:
        for phase, st in PHASES.items():
            rows = mv["rows"][mv["stages"] == st]
            if len(rows):
                refs[(mv["key"], phase)].append((mv["file"], int(rows[-1])))
    return refs, paths


def save(refs, paths, out):
    keys = sorted(refs)
    np.savez_compressed(
        out,
        keys=np.array([f"{k}:{p}" for k, p in keys]),
        offsets=np.cumsum([0] + [len(refs[k]) for k in keys]),
        refs=np.array([r for k in keys for r in refs[k]], dtype=np.int64),
        source_paths=np.array(paths),
    )


def load(path):
    """-> ({"AAAA>BAAA:descended": [(file, row), ...]}, source_paths)"""
    z = np.load(path)
    off, refs = z["offsets"], z["refs"]
    lib = {str(k): [tuple(map(int, r)) for r in refs[off[i] : off[i + 1]]] for i, k in enumerate(z["keys"])}
    return lib, [str(p) for p in z["source_paths"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True, help="the TRAINING split archive (play_train.npz)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    refs, paths = build(args.archive)
    save(refs, paths, args.out)
    transitions = {k for k, _ in refs}
    n = [len(v) for v in refs.values()]
    print(
        f"{len(refs)} (transition, phase) subgoals over {len(transitions)} transitions; exemplars per subgoal "
        f"min/median/max {min(n)}/{int(np.median(n))}/{max(n)} -> {args.out}"
    )


if __name__ == "__main__":
    main()

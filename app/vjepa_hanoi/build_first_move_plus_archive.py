# Build a larger one-move training split: the 48 first-move demos of expert6_first_move/train.npz plus every other
# recorded demo of the same six moves (ring 1 off a full stack).
#
#   python -m app.vjepa_hanoi.build_first_move_plus_archive \
#       --base /scratch/cw5167/checkpoints/vjepa2_ac_hanoi/data/expert6_first_move \
#       --sources sep15,sep10 --out /scratch/cw5167/checkpoints/vjepa2_ac_hanoi/data/first_move_plus_expert
#
# Sources (all /scratch/cw5167/datasets, same robot, camera crop and motion generator):
#   - sep15: hanoi_wm_roundtrip_20260915_223424 AAAA->CCCC / CCCC->AAAA, 50 + 50 expert episodes. Move 0 is
#     AAAA>BAAA / CCCC>BCCC. Full schema v4.
#   - sep10: hanoi_wm_roundtrip_20260910_195558, the same two directions, 50 + 50 expert episodes. This file has no
#     motion_stage, image_stale or image_repeated column, so they are rebuilt and checked against sep15:
#       * repeated = same image_timestamp_ns as the previous row of the episode (equal to sep15's stored flag on
#         every row);
#       * stale = command_monotonic_ns - image_receipt_monotonic_ns > 50 ms (on sep15 this flags every stored stale
#         row plus 0.3% more, so it only drops extra rows);
#       * stages: both sessions use fixed leg timing. Every sep15 move 0 has the same 480-row stage layout, and
#         sep10's move 0 has the same length and the same jaw-intent switch rows (156, 387 = grasp and release
#         starts), so sep15's layout is copied.
#   - play: hanoi_wm_20260924_210743 (the play recording of the non-expert column): every move of its 125 episodes
#     that is one of the six moves (91). Opt in only: it is the non-expert data's recording.
# Not used: the six-task expert files' episodes 8 and 9 (the val and test first moves). The base rows are taken
# verbatim from --base; for the new sources the same observation rule is applied (motion stages 1-9 of the move, image
# neither stale nor repeated). The rule is first checked by rebuilding the base split from its raw files.
#
# The training loader (hanoi.HanoiClipDataset) reads only source_paths, file_indices, source_observation_indices,
# segment_bounds, segment_kinds and motion_stages. Frames stay in the raw recordings. val/test are not rebuilt: the
# configs keep expert6_first_move/val.npz and test.npz.

import argparse
import hashlib
import itertools
import json
import os

import h5py
import numpy as np

BOARDS = ["".join(b) for b in itertools.product("ABC", repeat=4)]
H1_MOVES = ["AAAA>BAAA", "AAAA>CAAA", "BBBB>ABBB", "BBBB>CBBB", "CCCC>ACCC", "CCCC>BCCC"]
SEGMENT_KIND_MOVE = 4  # one expert move, as expert6_first_move
DATA = "/scratch/cw5167/datasets"
SOURCES = {
    "sep15": [f"{DATA}/hanoi_wm_roundtrip_20260915_223424_AAAA_to_CCCC.h5",
              f"{DATA}/hanoi_wm_roundtrip_20260915_223424_CCCC_to_AAAA.h5"],  # fmt: skip
    "sep10": [f"{DATA}/hanoi_wm_roundtrip_20260910_195558_AAAA_to_CCCC.h5",
              f"{DATA}/hanoi_wm_roundtrip_20260910_195558_CCCC_to_AAAA.h5"],  # fmt: skip
    "play": [f"{DATA}/hanoi_wm_20260924_210743.h5"],
}
SOURCE_TAGS = {"base": 0, "sep15": 1, "sep10": 2, "play": 3}
STALE_S = 0.05


def board_str(b):
    return "".join("ABC"[int(p)] for p in b)


def read_columns(h):
    """Per-row columns of one recording; rebuilt flags and stages where the file has none (sep10)."""
    off, ln = h["ep_offset"][:].astype(np.int64), h["ep_len"][:].astype(np.int64)
    c = {"off": off, "len": ln, "move": h["move_idx"][:].astype(np.int64), "board": h["board"][:],
         "success": h["episode_success"][:][off].astype(bool)}
    if "image_stale" in h:
        c["stale"], c["repeated"] = h["image_stale"][:].astype(bool), h["image_repeated"][:].astype(bool)
        c["stage"] = h["motion_stage"][:].astype(np.int64)
    else:
        ts = h["image_timestamp_ns"][:]
        rep = np.r_[False, ts[1:] == ts[:-1]]
        rep[off] = False
        c["repeated"] = rep
        c["stale"] = (h["command_monotonic_ns"][:] - h["image_receipt_monotonic_ns"][:]) / 1e9 > STALE_S
        c["stage"] = None
    return c


def moves_of_episode(c, o, n):
    """(move index, raw rows of motion stages 1-9) for each game move of one episode; stage-less files: move rows."""
    r = np.arange(o, o + n)
    out = []
    for m in np.unique(c["move"][r]):
        s = r[c["move"][r] == m]
        if c["stage"] is not None:
            s = s[(c["stage"][s] >= 1) & (c["stage"][s] <= 9)]
        if len(s):
            out.append((int(m), s))
    return out


def select(c, rows):
    return rows[~c["stale"][rows] & ~c["repeated"][rows]]


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="expert6_first_move (its train.npz rows are kept verbatim)")
    p.add_argument("--sources", default="sep15,sep10", help="comma list of " + ", ".join(SOURCES))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    sources = [s for s in args.sources.split(",") if s]
    assert set(sources) <= set(SOURCES), sources

    base = np.load(os.path.join(args.base, "train.npz"))
    base_meta = json.load(open(os.path.join(args.base, "metadata.json")))
    paths = [str(x) for x in base["source_paths"]]
    train_eps = base_meta["source"]["episode_split"]["train"]

    # 1. the observation rule reproduces the base split from its raw files (episodes 0-7, move 0)
    for fi, path in enumerate(paths):
        with h5py.File(path, "r") as h:
            c = read_columns(h)
        want = np.sort(base["source_observation_indices"][base["file_indices"] == fi])
        got = []
        for e in train_eps:
            m, s = moves_of_episode(c, c["off"][e], c["len"][e])[0]
            assert m == 0
            got.append(select(c, s))
        got = np.sort(np.concatenate(got))
        assert np.array_equal(got, want), (path, len(got), len(want))
    print(f"rule check: the base split's {len(base['source_observation_indices'])} rows rebuilt exactly", flush=True)

    cols = {k: [base[k]] for k in ("source_observation_indices", "file_indices", "motion_stages", "board_indices",
                                   "goal_board_indices", "goal_end_rows")}  # fmt: skip
    cols["segment_bounds"] = [base["segment_bounds"]]
    cols["episode_indices"] = [base["episode_indices"]]
    cols["source_tags"] = [np.zeros(len(base["file_indices"]), np.int64)]
    per_case = {t: {"base": 0} for t in H1_MOVES}
    for e_ in np.unique(base["episode_indices"]):
        at = np.nonzero(base["episode_indices"] == e_)[0][0]
        per_case[f"{BOARDS[base['board_indices'][at]]}>{BOARDS[base['goal_board_indices'][at]]}"]["base"] += 1

    # stage layout of a fixed-timing move 0 (sep15), for the stage-less sep10 files
    with h5py.File(SOURCES["sep15"][0], "r") as h:
        c15 = read_columns(h)
    m, s = moves_of_episode(c15, c15["off"][0], c15["len"][0])[0]
    layout15 = c15["stage"][s]
    layouts = {tuple(c15["stage"][moves_of_episode(c15, o, n)[0][1]]) for o, n in zip(c15["off"], c15["len"])}
    assert len(layouts) == 1, "sep15 move-0 stage layouts differ"
    with h5py.File(SOURCES["sep15"][0], "r") as h:
        jaw15 = h["action_abs"][s, 3]

    files_meta = []
    for src in sources:
        for path in SOURCES[src]:
            fi = len(paths)
            paths.append(path)
            with h5py.File(path, "r") as h:
                c = read_columns(h)
                jaw = h["action_abs"][:, 3]
                state, proprio = h["state"], h["proprio"]
                n_moves, n_rows = 0, 0
                for e, (o, n) in enumerate(zip(c["off"], c["len"])):
                    if not c["success"][e]:  # play episode 12 failed (the json manifest agrees)
                        continue
                    for m, s in moves_of_episode(c, o, n):
                        before, after = board_str(c["board"][s[0]]), board_str(c["board"][s[-1]])
                        mv = f"{before}>{after}"
                        if mv not in H1_MOVES:
                            continue
                        if c["stage"] is None:  # sep10: copy sep15's layout after checking the jaw-intent timing
                            assert len(s) == len(layout15) and np.array_equal(np.round(jaw[s], 3), np.round(jaw15, 3)), (path, e)
                            stage = layout15.copy()
                        else:
                            stage = c["stage"][s]
                            runs = [int(stage[0])] + [int(b) for a_, b in zip(stage[:-1], stage[1:]) if b != a_]
                            assert runs == list(range(1, 10)), (path, e, m, runs)
                        keep = ~c["stale"][s] & ~c["repeated"][s]
                        rows, stage = s[keep], stage[keep]
                        assert np.isfinite(state[rows[0] : rows[-1] + 1]).all() and np.isfinite(proprio[rows[0] : rows[-1] + 1]).all()
                        cols["source_observation_indices"].append(rows)
                        cols["file_indices"].append(np.full(len(rows), fi, np.int64))
                        cols["motion_stages"].append(stage.astype(base["motion_stages"].dtype))
                        cols["board_indices"].append(np.full(len(rows), BOARDS.index(before), np.int64))
                        cols["goal_board_indices"].append(np.full(len(rows), BOARDS.index(after), np.int64))
                        cols["goal_end_rows"].append(np.full(len(rows), rows[-1], np.int64))
                        cols["segment_bounds"].append(np.tile([rows[0], rows[-1] + 1], (len(rows), 1)))
                        # source tag * 1e8 + file * 1e6 + episode * 100 + move: unique per move, disjoint from base
                        cols["episode_indices"].append(np.full(len(rows), SOURCE_TAGS[src] * 10**8 + fi * 10**6 + e * 100 + m))
                        cols["source_tags"].append(np.full(len(rows), SOURCE_TAGS[src], np.int64))
                        per_case[mv][src] = per_case[mv].get(src, 0) + 1
                        n_moves += 1
                        n_rows += len(rows)
            files_meta.append({"source": src, "path": path, "moves": n_moves, "rows": n_rows,
                               "size_bytes": os.path.getsize(path)})  # fmt: skip
            print(f"{src} {os.path.basename(path)}: {n_moves} moves, {n_rows} rows", flush=True)

    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["first_move"] = np.ones(len(out["file_indices"]), bool)
    out["segment_kinds"] = np.full(len(out["file_indices"]), SEGMENT_KIND_MOVE, np.int64)
    os.makedirs(args.out, exist_ok=True)
    fn = os.path.join(args.out, "train.npz")
    np.savez_compressed(fn, **out, split=np.array("train"), source_paths=np.array(paths),
                        h1_moves=np.array(H1_MOVES))  # fmt: skip
    moves = {t: sum(v.values()) for t, v in per_case.items()}
    meta = {
        "base": os.path.abspath(args.base),
        "sources": sources,
        "files": files_meta,
        "rule": "motion stages 1-9 of the move, image neither stale nor repeated; sep10 flags rebuilt (header)",
        "val_test": "unchanged: expert6_first_move val.npz / test.npz (episodes 8 and 9 of the six-task files)",
        "moves_per_case": per_case,
        "moves": moves,
        "train": {"rows": int(len(out["file_indices"])), "moves": int(sum(moves.values())), "sha256": sha256(fn)},
    }
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({"moves_per_case": per_case, "train": meta["train"]}, indent=1))


if __name__ == "__main__":
    main()

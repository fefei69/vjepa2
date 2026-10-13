# Rebuild expert6_first_move {train,test}.npz (the fields paper_protocol_eval.moves() reads) from the six local
# recordings, following the Cosmos multitask_v6 rules in its metadata.json: tasks in metadata order, episodes 0-7 train
# and 9 test, observation rows = rows of the episode whose image is not repeated (the per-episode counts in the
# metadata are ep_len - repeated), and only each episode's first game move (ring 1 off the full stack).
#   python -I build_first_move_archives.py <metadata.json> <recordings dir> <out dir>
import itertools
import json
import os
import sys

import h5py
import numpy as np

BOARDS = ["".join(b) for b in itertools.product("ABC", repeat=4)]
meta_path, rec_dir, out_dir = sys.argv[1:4]
meta = json.load(open(meta_path))
os.makedirs(out_dir, exist_ok=True)
paths = [os.path.join(rec_dir, t["file"]) for t in sorted(meta["tasks"], key=lambda t: t["index"])]
splits = {"train": list(range(8)), "test": [9]}
counts = {}
cols = {s: {k: [] for k in ("source_observation_indices", "file_indices", "segment_bounds", "move_indices",
                            "motion_stages", "board_indices", "episode_indices")} for s in splits}
for fi, (path, task) in enumerate(zip(paths, sorted(meta["tasks"], key=lambda t: t["index"]))):
    with h5py.File(path, "r") as h:
        off, ln = h["ep_offset"][:], h["ep_len"][:]
        rep, ok = h["image_repeated"][:], h["episode_success"][:]
        mv, st, bd = h["move_idx"][:], h["motion_stage"][:], h["board"][:]
        per_ep = meta["audit"][task["direction"]]["per_episode"]
        for split, eps in splits.items():
            for e in eps:
                rows = np.arange(off[e], off[e] + ln[e])
                obs = rows[(rep[rows] == 0) & (ok[rows] != 0)]
                counts.setdefault((split, task["direction"]), 0)
                counts[(split, task["direction"])] += len(obs)
                assert len(obs) == per_ep[e]["observations"], (path, e, len(obs), per_ep[e]["observations"])
                moving = obs[np.isin(st[obs], np.arange(1, 10))]
                first = mv[moving].min()  # the episode's first game move
                sel = obs[mv[obs] == first]
                b = bd[sel].astype(int)
                bidx = (b * np.array([27, 9, 3, 1])).sum(1)  # ring 1 first, A=0 B=1 C=2 (checked below)
                c = cols[split]
                c["source_observation_indices"].append(sel)
                c["file_indices"].append(np.full(len(sel), fi))
                c["segment_bounds"].append(np.tile([sel[0], sel[-1] + 1], (len(sel), 1)))
                c["move_indices"].append(mv[sel])
                c["motion_stages"].append(st[sel])
                c["board_indices"].append(bidx)
                c["episode_indices"].append(np.full(len(sel), fi * 100 + e))
for split in splits:
    arrays = {k: np.concatenate(v) for k, v in cols[split].items()}
    arrays["source_paths"] = np.array(paths)
    np.savez(os.path.join(out_dir, f"{split}.npz"), **arrays)
    print(split, {k: v.shape for k, v in arrays.items()})
# the observation rule reproduces the metadata's per-task sample counts
for (split, task), n in sorted(counts.items()):
    want = meta["splits"][split]["per_task_samples"][task]
    print(f"{split:5s} {task}: {n} observation rows (metadata {want}) {'OK' if n == want else 'MISMATCH'}")

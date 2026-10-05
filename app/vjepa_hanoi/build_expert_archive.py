# Build V-JEPA 2-AC split archives for the six-task expert data from the Cosmos multitask_v6 archives.
#
#   python -m app.vjepa_hanoi.build_expert_archive \
#       --cosmos /scratch/cw5167/workspace/cosmos-policy/data/hanoi_cosmos/multitask_v6 \
#       --out /scratch/cw5167/checkpoints/vjepa2_ac_hanoi/data/expert6_multitask_v6
#
# Same data as the Cosmos six-task run: the six directed full-stack recordings (one file per task, ten successful
# expert episodes each), episodes 0-7 of every file train, 8 validate, 9 test, and the Cosmos observation rule (every
# row of a successful episode whose image is neither stale nor repeated). The observation rows are taken verbatim
# from the Cosmos archives (`source_observation_indices`); nothing is re-drawn. Each episode is one segment, so a
# training clip never crosses an episode boundary. The Cosmos tree is only read.
#
# Output: {train,val,test}.npz in the format app/vjepa_hanoi/hanoi.py reads (the openpi play_k5 archive layout),
# with the per-row motion stage, move index and board read from the raw recordings, plus metadata.json (provenance).

import argparse
import hashlib
import json
import os

import h5py
import numpy as np

SEGMENT_KIND_EXPERT = 3  # 0 walk, 1 crop, 2 clip in the play archives; 3 = a whole expert episode


def board_index(board):
    """Peg per ring (0 A, 1 B, 2 C), ring 1 first -> index in itertools.product('ABC', repeat=4) order."""
    board = np.asarray(board, dtype=np.int64)
    return board[:, 0] * 27 + board[:, 1] * 9 + board[:, 2] * 3 + board[:, 3]


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cosmos", required=True, help="cosmos-policy data/hanoi_cosmos/multitask_v6 directory")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    meta = json.load(open(os.path.join(args.cosmos, "metadata.json")))
    files = sorted(meta["files"], key=lambda f: f["task_index"])
    assert [f["task_index"] for f in files] == list(range(len(files)))
    paths = [f["path"] for f in files]
    for f in files:  # the recordings must be the ones the Cosmos archive was built from
        st = os.stat(f["path"])
        unchanged = st.st_size == f["size_bytes"] and st.st_mtime_ns == f["mtime_ns"]
        assert unchanged, f"{f['path']} changed since the Cosmos build"

    raw = {}
    for i, path in enumerate(paths):
        with h5py.File(path, "r") as h:
            raw[i] = {
                "stage": h["motion_stage"][:].astype(np.int64),
                "move": h["move_idx"][:].astype(np.int64),
                "board": board_index(h["board"][:]),
                "stale": h["image_stale"][:] > 0,
                "repeated": h["image_repeated"][:] > 0,
                "offsets": h["ep_offset"][:].astype(np.int64),
                "lengths": h["ep_len"][:].astype(np.int64),
            }

    os.makedirs(args.out, exist_ok=True)
    report = {}
    for split in ("train", "val", "test"):
        src = os.path.join(args.cosmos, f"{split}.npz")
        z = np.load(src)
        rows, fi = z["source_observation_indices"].astype(np.int64), z["file_indices"].astype(np.int64)
        episodes = z["episode_indices"].astype(np.int64)  # task index * 100 + episode within the file
        bounds = np.zeros((len(rows), 2), np.int64)
        stage, move, board = (np.zeros(len(rows), np.int64) for _ in range(3))
        for i in np.unique(fi):
            sel = fi == i
            r, ep = rows[sel], episodes[sel] % 100
            lo = raw[i]["offsets"][ep]
            bounds[sel] = np.stack([lo, lo + raw[i]["lengths"][ep]], 1)
            assert np.all((r >= bounds[sel, 0]) & (r < bounds[sel, 1])), "row outside its episode"
            assert not np.any(raw[i]["stale"][r] | raw[i]["repeated"][r]), "row violates the observation rule"
            stage[sel], move[sel], board[sel] = raw[i]["stage"][r], raw[i]["move"][r], raw[i]["board"][r]
        if "source_episode_bounds" in z.files:  # train archive carries Cosmos's own episode bounds: must agree
            assert np.array_equal(z["source_episode_bounds"], bounds)
        # every non-stale, non-repeated row of each listed episode is present exactly once (Cosmos observation rule)
        for i in np.unique(fi):
            for ep in np.unique(episodes[fi == i] % 100):
                lo, hi = raw[i]["offsets"][ep], raw[i]["offsets"][ep] + raw[i]["lengths"][ep]
                expect = lo + np.nonzero(~(raw[i]["stale"][lo:hi] | raw[i]["repeated"][lo:hi]))[0]
                got = np.sort(rows[(fi == i) & (episodes % 100 == ep)])
                assert np.array_equal(expect, got), f"file {i} episode {ep}: rows differ from the observation rule"
        out = os.path.join(args.out, f"{split}.npz")
        np.savez_compressed(
            out,
            split=np.array(split),
            source_paths=np.array(paths),
            file_indices=fi,
            source_observation_indices=rows,
            episode_indices=episodes,
            segment_bounds=bounds,
            segment_kinds=np.full(len(rows), SEGMENT_KIND_EXPERT, np.int64),
            motion_stages=stage,
            move_indices=move,
            board_indices=board,
            task_indices=z["task_indices"].astype(np.int64),
        )
        report[split] = {
            "rows": int(len(rows)),
            "episodes": sorted(map(int, np.unique(episodes))),
            "cosmos_samples": meta["splits"][split]["samples"],
            "cosmos_archive_sha256": sha256(src),
            "cosmos_archive_sha256_recorded": meta["splits"][split]["sha256"],
            "sha256": sha256(out),
        }
        assert report[split]["rows"] == report[split]["cosmos_samples"]
        assert report[split]["cosmos_archive_sha256"] == report[split]["cosmos_archive_sha256_recorded"]
        print(f"{split}: {len(rows)} rows, episodes {report[split]['episodes']} -> {out}", flush=True)

    assert not set(report["train"]["episodes"]) & (set(report["val"]["episodes"]) | set(report["test"]["episodes"]))
    assert not set(report["val"]["episodes"]) & set(report["test"]["episodes"])
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(
            {
                "source": os.path.abspath(args.cosmos),
                "contract": meta["contract"],
                "observation_rule": meta["observation_rule"],
                "episode_split": meta["episode_split"],
                "files": files,
                "segment_kind": f"{SEGMENT_KIND_EXPERT} = one whole expert episode",
                "board_index": "itertools.product('ABC', repeat=4) order, ring 1 first (AAAA = 0 ... CCCC = 80)",
                "splits": report,
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()

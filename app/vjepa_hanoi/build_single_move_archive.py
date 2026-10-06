# Build the one-move dataset: every game move of the six-task expert recordings as its own episode.
#
#   python -m app.vjepa_hanoi.build_single_move_archive \
#       --expert /scratch/cw5167/checkpoints/vjepa2_ac_hanoi/data/expert6_multitask_v6 \
#       --cosmos /scratch/cw5167/workspace/cosmos-policy/data/hanoi_cosmos/multitask_v6 \
#       --play_archive /scratch/cw5167/workspace/openpi/data/hanoi/play_k5_pi05/indices/play_val.npz \
#       --out /scratch/cw5167/checkpoints/vjepa2_ac_hanoi/data/expert6_single_move
#
# Source: the Cosmos six-task split, rebuilt by build_expert_archive.py. It has six directed full-stack recordings
# with ten successful episodes each: episodes 0-7 train, 8 validation, 9 test. Rows follow the Cosmos observation
# rule, taken verbatim. Each complete game move (motion stages 1-9 in order) becomes one episode. Its goal is the board
# after the move, and its goal frame is the move's last usable row (the retreat row, which already reads the new board).
# Rows outside any move (episode start, return and hold) are dropped and counted. The six test cases are the first
# move of each recording: ring 1 off a full stack (AAAA->BAAA, AAAA->CAAA, BBBB->ABBB, BBBB->CBBB, CCCC->ACCC,
# CCCC->BCCC), flagged `first_move`. --first_moves_only keeps only those six moves: 8 train / 1 val / 1 test demos each.
#
# Columns and labels follow the play_k5 handover (openpi/docs/hanoi_play_k5_dataset_handover.md, sections 5-7), so
# the pi0.5 / Cosmos loaders read it unchanged:
#   - states = [joint_positions, proprio[:, 6]];
#   - labels: slot j = 1..16 is raw row t + 3j, with XYZ = reference_pose[:, 0:3] and jaw = action_abs[:, 3]; slots
#     past the goal row take that row and are marked padded;
#   - goal = the sentence of the after-move board (the 81 handover prompts).
# The rules are checked against the stored play archive. Every state and every unpadded label is checked equal to
# the Cosmos six-task archive row for row.

import argparse
import hashlib
import itertools
import json
import os
from collections import deque

import h5py
import numpy as np

BOARDS = ["".join(b) for b in itertools.product("ABC", repeat=4)]
HORIZON, FRAMESKIP = 16, 3
SEGMENT_KIND_MOVE = 4  # 0 walk, 1 crop, 2 clip (play), 3 whole expert episode, 4 one expert move
PROMPTS_SHA256 = "ac402b1d646c8d99edc0d5432b9056b95ac6043a82a73adb1a7a7670404ad64a"


def prompt_for_board(board):
    """The handover's goal sentence (openpi hanoi_play_policy.prompt_for_board, character for character)."""
    clauses = []
    for peg in "ABC":
        rings = [str(i + 1) for i in range(4) if board[i] == peg]
        if not rings:
            clauses.append(f"peg {peg} is empty")
        elif len(rings) == 1:
            clauses.append(f"peg {peg} holds ring {rings[0]}")
        else:
            clauses.append(f"peg {peg} holds rings " + ", ".join(rings[:-1]) + " and " + rings[-1])
    return "Goal: " + ", ".join(clauses) + "."


def legal_moves(b):
    for ring, peg in enumerate(b):
        if any(b[r] == peg for r in range(ring)):
            continue
        for t in "ABC":
            if t != peg and not any(b[r] == t for r in range(ring)):
                yield b[:ring] + t + b[ring + 1 :]


def graph_distance(a, b):
    dist, q = {a: 0}, deque([a])
    while q:
        x = q.popleft()
        if x == b:
            return dist[x]
        for n in legal_moves(x):
            if n not in dist:
                dist[n] = dist[x] + 1
                q.append(n)
    raise ValueError((a, b))


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def labels(raw, rows, goal_end):
    """Section-5 chunks for observation rows `rows` whose goal frame is `goal_end` (same length)."""
    slots = rows[:, None] + FRAMESKIP * np.arange(1, HORIZON + 1)[None]
    pad = slots > goal_end[:, None]
    src = np.where(pad, goal_end[:, None], slots)
    act = np.concatenate([raw["reference_pose"][src][..., :3], raw["action_abs"][src][..., 3:4]], -1)
    return act.astype(np.float32), pad, src


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", required=True, help="expert6_multitask_v6 archives (build_expert_archive.py)")
    p.add_argument("--cosmos", required=True, help="Cosmos multitask_v6 archives (read only, for the cross-check)")
    p.add_argument("--play_archive", required=True, help="a play_k5 handover archive: prompts and rule check")
    p.add_argument("--out", required=True)
    p.add_argument("--first_moves_only", action="store_true", help="keep only the six test-case moves (first moves)")
    args = p.parse_args()

    prompts = [prompt_for_board(b) for b in BOARDS]
    assert hashlib.sha256("\n".join(prompts).encode()).hexdigest() == PROMPTS_SHA256
    play = np.load(args.play_archive)
    assert list(play["prompts"]) == prompts
    # the section-5 rules, re-checked on stored play rows (expert file: file index 1)
    sel = np.nonzero(play["file_indices"] == 1)[0][::97][:200]
    with h5py.File(str(play["source_paths"][1]), "r") as h:
        raw = {k: h[k][:] for k in ("reference_pose", "action_abs")}
    a, pad, src = labels(raw, play["source_observation_indices"][sel], play["goal_end_rows"][sel])
    assert np.allclose(a, play["actions"][sel]) and np.array_equal(pad, play["actions_is_pad"][sel])
    assert np.array_equal(src, play["source_action_indices"][sel])

    meta_in = json.load(open(os.path.join(args.expert, "metadata.json")))
    paths = [f["path"] for f in meta_in["files"]]
    raws = []
    for path in paths:
        with h5py.File(path, "r") as h:
            raws.append({k: h[k][:] for k in ("reference_pose", "action_abs", "joint_positions", "proprio")})

    os.makedirs(args.out, exist_ok=True)
    report = {}
    for split in ("train", "val", "test"):
        z = np.load(os.path.join(args.expert, f"{split}.npz"))
        cz = np.load(os.path.join(args.cosmos, f"{split}.npz"))
        assert np.array_equal(z["source_observation_indices"], cz["source_observation_indices"])
        obs, fi, ep = z["source_observation_indices"], z["file_indices"], z["episode_indices"]
        mv, st, bd = z["move_indices"], z["motion_stages"], z["board_indices"]
        keep = np.zeros(len(obs), bool)
        seg_lo, goal_end, goal_board, first = (np.zeros(len(obs), np.int64) for _ in range(4))
        n_moves, moves_seen = 0, []
        for e in np.unique(ep):
            in_ep = np.nonzero(ep == e)[0]
            in_ep = in_ep[np.argsort(obs[in_ep])]
            real = [m for m in np.unique(mv[in_ep]) if np.isin(st[in_ep][mv[in_ep] == m], np.arange(1, 10)).any()]
            for k, m in enumerate(sorted(real)):
                s = in_ep[(mv[in_ep] == m) & np.isin(st[in_ep], np.arange(1, 10))]
                runs = [int(st[s[0]])] + [int(b) for a_, b in zip(st[s][:-1], st[s][1:]) if b != a_]
                assert runs == list(range(1, 10)), (split, e, m, runs)
                assert np.all(np.diff(obs[s]) > 0)
                before, after = BOARDS[bd[s[0]]], BOARDS[bd[s[-1]]]
                assert after in set(legal_moves(before)), (before, after)
                keep[s] = k == 0 or not args.first_moves_only
                seg_lo[s], goal_end[s], goal_board[s], first[s] = obs[s[0]], obs[s[-1]], bd[s[-1]], k == 0
                if k == 0 or not args.first_moves_only:
                    n_moves += 1
                    moves_seen.append((int(e), k, f"{before}>{after}"))
        idx = np.nonzero(keep)[0]
        rows, files = obs[idx], fi[idx]
        cols = {k: [] for k in ("states", "actions", "actions_is_pad", "source_action_indices", "measured_xyz",
                                "speed_m_s", "observation_jaw_intent")}  # fmt: skip
        for f in np.unique(files):
            at = np.nonzero(files == f)[0]
            raw, r = raws[f], rows[at]
            a, pad, src = labels(raw, r, goal_end[idx][at])
            states = np.concatenate([raw["joint_positions"][r], raw["proprio"][r, 6:7]], 1).astype(np.float32)
            xyz = raw["proprio"][r, :3]
            # backward difference at 30 Hz; an episode's first row uses the forward one (the Cosmos archive's rule)
            starts = r == z["segment_bounds"][idx[at], 0]
            nb = np.where(starts, r + 1, r - 1)
            speed = (np.linalg.norm(xyz - raw["proprio"][nb, :3], axis=1) * 30).astype(np.float32)
            for k, v in (("states", states), ("actions", a), ("actions_is_pad", pad), ("source_action_indices", src),
                         ("measured_xyz", xyz.astype(np.float32)), ("speed_m_s", speed),
                         ("observation_jaw_intent", raw["action_abs"][r, 3].astype(np.float32))):  # fmt: skip
                cols[k].append((at, v))
        out = {}
        for k, parts in cols.items():
            v0 = parts[0][1]
            arr = np.zeros((len(idx),) + v0.shape[1:], v0.dtype)
            for at, v in parts:
                arr[at] = v
            out[k] = arr
        # cross-check against the Cosmos six-task archive, row for row
        assert np.allclose(out["states"], cz["states"][idx])
        unpadded = ~out["actions_is_pad"]
        assert np.allclose(out["actions"][unpadded], cz["actions"][idx][unpadded])
        assert np.array_equal(out["source_action_indices"][unpadded], cz["source_action_indices"][idx][unpadded])
        assert np.allclose(out["measured_xyz"], cz["cartesian_positions"][idx])
        assert np.allclose(out["speed_m_s"], cz["measured_speed_m_per_s"][idx], atol=1e-5)
        cur = bd[idx]
        dist = {(c, g): graph_distance(BOARDS[c], BOARDS[g]) for c, g in set(zip(cur.tolist(), goal_board[idx].tolist()))}
        move_id = ep[idx] * 100 + mv[idx]  # task * 10000 + episode * 100 + move
        np.savez_compressed(
            os.path.join(args.out, f"{split}.npz"),
            source_observation_indices=rows,
            **out,
            episode_indices=move_id,
            source_episode_bounds=np.stack([seg_lo[idx], goal_end[idx] + 1], 1),
            task_indices=z["task_indices"][idx],
            file_indices=files,
            prompt_indices=goal_board[idx],
            segment_bounds=np.stack([seg_lo[idx], goal_end[idx] + 1], 1),
            segment_kinds=np.full(len(idx), SEGMENT_KIND_MOVE, np.int64),
            move_indices=mv[idx],
            motion_stages=st[idx],
            board_indices=cur,
            goal_end_rows=goal_end[idx],
            goal_board_indices=goal_board[idx],
            goal_moves_ahead=np.zeros(len(idx), np.int64),
            goal_graph_distance=np.array([dist[(c, g)] for c, g in zip(cur.tolist(), goal_board[idx].tolist())]),
            next_board_indices=goal_board[idx],
            first_move=first[idx].astype(bool),
            source_expert_episode_indices=ep[idx],
            validated=np.array(True),
            state_columns=np.arange(7),
            horizon=np.array(HORIZON),
            frameskip=np.array(FRAMESKIP),
            split=np.array(split),
            source_paths=np.array(paths),
            prompts=np.array(prompts),
        )
        firsts = sorted({t for e, k, t in moves_seen if k == 0})
        report[split] = {
            "rows": int(len(idx)),
            "rows_outside_moves_dropped": int(len(obs) - len(idx)),
            "moves": n_moves,
            "distinct_transitions": len({t for _, _, t in moves_seen}),
            "first_moves": firsts,
            "first_move_rows": int(first[idx].sum()),
            "padded_slot_fraction": float(out["actions_is_pad"].mean()),
            "sha256": sha256(os.path.join(args.out, f"{split}.npz")),
        }
        print(split, json.dumps(report[split]), flush=True)
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(
            {
                "source": {"expert_archive": os.path.abspath(args.expert), "cosmos": os.path.abspath(args.cosmos),
                           "files": meta_in["files"], "episode_split": meta_in["episode_split"]},  # fmt: skip
                "episode": "one complete game move (motion stages 1-9) of an expert episode",
                "first_moves_only": args.first_moves_only,
                "goal": "board after the move; goal frame = the move's last usable row",
                "test_cases": "first move of each recording (first_move = True): ring 1 off a full stack",
                "labels": "handover section 5: slot j = raw row t + 3j, reference_pose XYZ + action_abs jaw, cut at "
                "goal_end_rows (padded slots repeat it)",
                "prompts_sha256": PROMPTS_SHA256,
                "segment_kind": f"{SEGMENT_KIND_MOVE} = one expert move",
                "episode_indices": "task * 10000 + expert episode * 100 + move index (unique per move)",
                "splits": report,
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()

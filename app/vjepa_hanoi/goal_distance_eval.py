# Where does plain V-JEPA 2-AC planning work? Steering accuracy as a function of goal distance.
#
#   python -m app.vjepa_hanoi.goal_distance_eval --fname <yaml> --checkpoint <best.pt> --goal_type move_end_1 \
#       --archive <openpi play_test.npz> --max_rows 400
#
# Plain baseline: the repo planner (notebooks/utils/mpc_utils.py::cem) with its default WorldModel settings (rollout 2,
# 400 samples, top-10, 10 iterations, maxnorm 0.05 m), a single real goal image, no subgoals, no solver, no fixed axes,
# no step cap. From a usable held-out row r inside a ring transfer (motion stages 1-9), the goal is the recorded frame:
#   step_k       k model steps later (k x 0.267 s), k in 1, 2, 4, 8, 16
#   stage_end    the end of the current motion stage
#   move_end_j   the end of the j-th game move counting the current one (j = 1: the board after this move; 3, 7, 15)
# The first planned action is compared with the expert's actual next step (row r -> r + 8). The same rows are used for
# every goal type (those whose goal exists in the walk), so the goal types are directly comparable.

import argparse
import json
import time
from collections import defaultdict

import h5py
import numpy as np
import torch
import yaml

from app.vjepa_droid.transforms import make_transforms
from app.vjepa_hanoi.ac_step import encode_clips
from app.vjepa_hanoi.eval import STAGES, build_models, strip_prefixes
from app.vjepa_hanoi.hanoi import hanoi_states, poses_to_diffs
from app.vjepa_hanoi.plan_eval import load_mpc_utils, make_step_fn

GOAL_TYPES = ["step_1", "step_2", "step_4", "step_8", "step_16", "stage_end"] + [
    f"move_end_{j}" for j in (1, 3, 7, 15)
]
FSTP = 8  # raw rows per model step (30 Hz, fps 4)


def candidate_rows(archive, real_moves_only=True):
    """Usable rows in transfer stages with every goal row they admit (rows are raw indices in their file).

    move_end_j counts real game moves only: moves that contain a transfer stage (1-9). The terminal hold/padding
    row at the end of a walk or episode carries the next move index but is not a move (real_moves_only=False
    reproduces the first sweep, which counted it)."""
    z = np.load(archive)
    obs, fi, seg = z["source_observation_indices"], z["file_indices"], z["segment_bounds"]
    move, stage = z["move_indices"], z["motion_stages"]
    paths = [str(p) for p in z["source_paths"]]
    out = []
    for key in np.unique(np.stack([fi, seg[:, 0]], 1), axis=0):
        sel = np.nonzero((fi == key[0]) & (seg[:, 0] == key[1]))[0]
        sel = sel[np.argsort(obs[sel])]
        rows, mv, st = obs[sel], move[sel], stage[sel]
        usable = set(rows.tolist())
        moves = [m for m in np.unique(mv) if not real_moves_only or np.isin(st[mv == m], np.arange(1, 10)).any()]
        move_end = {m: rows[mv == m].max() for m in moves}
        for i, r in enumerate(rows):
            if st[i] not in range(1, 10) or r + FSTP not in usable:
                continue
            goals = {f"step_{k}": r + FSTP * k for k in (1, 2, 4, 8, 16) if r + FSTP * k in usable}
            j = i
            while j + 1 < len(rows) and st[j + 1] == st[i] and mv[j + 1] == mv[i]:
                j += 1
            if rows[j] > r:
                goals["stage_end"] = rows[j]
            for n in (1, 3, 7, 15):
                if mv[i] + n - 1 in move_end:
                    goals[f"move_end_{n}"] = move_end[mv[i] + n - 1]
            out.append({"file": int(key[0]), "row": int(r), "stage": int(st[i]), "goals": goals})
    return out, paths


def summarize(rec, m):
    """Steering metrics over the rows selected by mask m (y-z plane; the arm moves in it)."""
    t, pl = rec["true"][m, 1:3] * 1e3, rec["plan"][m, 1:3] * 1e3  # y-z: the plane the arm moves in
    mag = np.linalg.norm(t, axis=1)
    mov = mag >= 5
    cos = np.sum(t[mov] * pl[mov], 1) / (mag[mov] * np.linalg.norm(pl[mov], axis=1) + 1e-9)
    tg, pg = rec["true"][m, 6], rec["plan"][m, 6]
    ev = np.abs(tg) >= 0.25
    nan = float("nan")
    return {
        "rows": int(m.sum()),
        "goal_seconds_median": float(np.median(rec["goal_seconds"][m])) if m.any() else nan,
        "moving_rows": int(mov.sum()),
        "direction_cos_median": float(np.median(cos)) if mov.any() else nan,
        "direction_correct": float(np.mean(cos > 0)) if mov.any() else nan,
        "direction_within_45deg": float(np.mean(cos > np.cos(np.pi / 4))) if mov.any() else nan,
        "yz_err_over_zero_median": (
            float(np.median(np.linalg.norm(pl[mov] - t[mov], axis=1) / mag[mov])) if mov.any() else nan
        ),
        "gripper_events": int(ev.sum()),
        "gripper_sign_accuracy": float(np.mean(np.sign(pg[ev]) == np.sign(tg[ev]))) if ev.any() else nan,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fname", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--archive", required=True, help="held-out split archive (play_test.npz)")
    p.add_argument("--goal_type", choices=GOAL_TYPES, required=True)
    p.add_argument("--max_rows", type=int, default=400)
    p.add_argument("--rollout", type=int, default=2, help="repo WorldModel default")
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--cem_steps", type=int, default=10)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.fname) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    d = cfg["data"]
    crop = d.get("crop_size", 256)
    tpf = (crop // d["patch_size"]) ** 2
    norm = cfg["loss"]["normalize_reps"]
    dtype = torch.bfloat16 if str(cfg["meta"].get("dtype")).lower() == "bfloat16" else torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder, predictor = build_models(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    encoder.load_state_dict(strip_prefixes(ckpt["target_encoder"]), strict=True)
    predictor.load_state_dict(strip_prefixes(ckpt["predictor"]), strict=True)
    del ckpt
    encoder.eval()
    predictor.eval()
    mpc = load_mpc_utils()
    step_fn = make_step_fn(predictor, tpf, norm, mpc.compute_new_pose)
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop,
    )
    cem_kw = dict(rollout=args.rollout, samples=args.samples, topk=10, cem_steps=args.cem_steps,
                  momentum_mean=0.15, momentum_std=0.15, maxnorm=0.05)  # fmt: skip

    cands, paths = candidate_rows(args.archive)
    pick = np.linspace(0, len(cands) - 1, min(args.max_rows, len(cands))).astype(int)  # same rows for every goal type
    files = [h5py.File(pth, "r") for pth in paths]
    rec = defaultdict(list)
    t0 = time.time()
    for n, i in enumerate(pick):
        c = cands[i]
        if args.goal_type not in c["goals"]:
            continue
        h, r, g = files[c["file"]], c["row"], int(c["goals"][args.goal_type])
        pose = h["state"][[r, r + FSTP]].astype(np.float64)
        jaw = h["proprio"][[r, r + FSTP], 6]
        s7 = hanoi_states(pose, np.repeat(jaw[:, None], 8, axis=1))
        true = poses_to_diffs(s7)[0]
        clip = transform(np.stack([h["pixels"][r], h["pixels"][g]]))[None].to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            z = encode_clips(encoder, clip, norm).float()
            z0, zg = z[:, :tpf].view(1, 1, tpf, -1), z[:, tpf:].view(1, 1, tpf, -1)
            pose_t = torch.as_tensor(s7[None, :1], dtype=torch.float32, device=device)
            torch.manual_seed(n)
            plan = mpc.cem(context_frame=z0, context_pose=pose_t, goal_frame=zg, world_model=step_fn, **cem_kw)[0]
        rec["true"].append(true)
        rec["plan"].append(plan[0].float().cpu().numpy())
        rec["stage"].append(c["stage"])
        rec["goal_seconds"].append((g - r) / 30.0)
        rec["file_index"].append(c["file"])
        rec["row"].append(r)
        if len(rec["true"]) % 50 == 0:
            print(f"{len(rec['true'])} rows, {(time.time() - t0) / len(rec['true']):.2f} s/row", flush=True)
    rec = {k: np.asarray(v) for k, v in rec.items()}
    np.savez_compressed(args.out.replace(".json", "_rows.npz"), **rec)

    allm = np.ones(len(rec["stage"]), bool)
    result = {
        "goal_type": args.goal_type,
        "checkpoint": args.checkpoint,
        "planner": cem_kw | {"source": "notebooks/utils/mpc_utils.py::cem (defaults, no aids)"},
        "overall": summarize(rec, allm),
        "by_stage": {STAGES[int(s)]: summarize(rec, rec["stage"] == s) for s in np.unique(rec["stage"])},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    o = result["overall"]
    print(
        f"[{args.goal_type}] {o['rows']} rows, goal {o['goal_seconds_median']:.1f} s ahead (median): direction "
        f"correct {o['direction_correct']:.2f}, within 45 deg {o['direction_within_45deg']:.2f}, cos "
        f"{o['direction_cos_median']:.2f}, gripper {o['gripper_sign_accuracy']:.2f} ({o['gripper_events']})"
        f" -> {args.out}"
    )


if __name__ == "__main__":
    main()

# Is the goal-image energy monotone along the expert's own path? (frozen encoder only; no predictor, no planner)
#
#   python -m app.vjepa_hanoi.energy_profile --fname <yaml> --checkpoint <ckpt with target_encoder> \
#       --archive <split archive .npz> --out <json>
#
# The plain planner chooses actions that lower E = mean |z - z_goal| between predicted and goal latents (layer-normed
# encoder features). The encoder is frozen, so along a REAL trajectory E depends only on the frames, never on what
# the predictor was trained on: a perfect predictor would reproduce exactly these values. If the expert's next step
# raises E, a planner that looks only a few steps ahead cannot justify following the expert there, whatever data the
# predictor saw.
#
# For every real game move (moves containing a transfer stage 1-9) the expert path is sampled every 8 raw rows
# (one model step, 0.267 s). For each sample k:
#   uphill        E_{k+1} > E_k: the expert's next step moves away from the goal in latent space
#   barrier_k     steps until the expert path first gets below E_k (the lookahead a planner needs to see progress)
# Goals: the end of the move (one game move), the end of the current motion stage (subgoal), and for whole expert
# episodes the end of the episode (15 moves). The repo planner's default rollout is 2 steps.

import argparse
import json
from collections import defaultdict

import h5py
import numpy as np
import torch
import yaml

from app.vjepa_droid.transforms import make_transforms
from app.vjepa_hanoi.ac_step import encode_clips
from app.vjepa_hanoi.eval import STAGES, build_models, strip_prefixes

FSTP = 8


def barrier(e):
    """For each k, the number of steps until e first drops strictly below e[k] (len(e) if never)."""
    out = np.full(len(e), np.inf)
    for k in range(len(e)):
        later = np.nonzero(e[k + 1 :] < e[k])[0]
        if len(later):
            out[k] = later[0] + 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fname", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rollout", type=int, default=2, help="planner lookahead to compare barriers with")
    p.add_argument("--max_segments", type=int, default=None, help="for quick tests")
    args = p.parse_args()

    with open(args.fname) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    d = cfg["data"]
    crop = d.get("crop_size", 256)
    tpf = (crop // d["patch_size"]) ** 2
    norm = cfg["loss"]["normalize_reps"]
    dtype = torch.bfloat16 if str(cfg["meta"].get("dtype")).lower() == "bfloat16" else torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder, _ = build_models(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    encoder.load_state_dict(strip_prefixes(ckpt["target_encoder"]), strict=True)
    del ckpt
    encoder.eval()
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop,
    )

    @torch.no_grad()
    def encode(images, chunk=64):
        outs = []
        for i in range(0, len(images), chunk):
            clip = transform(np.stack(images[i : i + chunk]))[None].to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                z = encode_clips(encoder, clip, norm).float()
            outs.append(z.view(-1, tpf, z.size(-1)).cpu())
        return torch.cat(outs)

    z = np.load(args.archive)
    obs, fi, seg = z["source_observation_indices"], z["file_indices"], z["segment_bounds"]
    move, stage, kinds = z["move_indices"], z["motion_stages"], z["segment_kinds"]
    files = [h5py.File(str(pth), "r") for pth in z["source_paths"]]

    stats = defaultdict(lambda: defaultdict(list))  # goal type -> stage -> list of (uphill, barrier)
    episode_profiles = []
    for key in np.unique(np.stack([fi, seg[:, 0]], 1), axis=0)[: args.max_segments]:
        sel = np.nonzero((fi == key[0]) & (seg[:, 0] == key[1]))[0]
        sel = sel[np.argsort(obs[sel])]
        rows, mv, st = obs[sel], move[sel], stage[sel]
        usable = {int(r): i for i, r in enumerate(rows)}
        h = files[int(key[0])]
        real_moves = [m for m in np.unique(mv) if np.isin(st[mv == m], np.arange(1, 10)).any()]
        ep_rows, ep_stage = [], []
        for m in real_moves:
            mrows = rows[mv == m]
            # the expert path at one model step per sample, snapped to the nearest usable row (within 3 rows)
            path = []
            for r in range(int(mrows[0]), int(mrows[-1]) + 1, FSTP):
                near = [r + o for o in (0, 1, -1, 2, -2, 3, -3) if r + o in usable and mrows[0] <= r + o <= mrows[-1]]
                if near:
                    path.append(near[0])
            if path[-1] != int(mrows[-1]):
                path.append(int(mrows[-1]))  # the move's last usable row: the one-move goal
            pst = np.array([st[usable[r]] for r in path])
            # stage-end goal per sample: last usable row of the current (move, stage) run
            stage_end = {}
            for r, s in zip(path, pst):
                i = usable[r]
                j = i
                while j + 1 < len(rows) and st[j + 1] == s and mv[j + 1] == m:
                    j += 1
                stage_end[r] = int(rows[j])
            need = sorted(set(path) | set(stage_end.values()))
            lat = encode([h["pixels"][r] for r in need])
            idx = {r: i for i, r in enumerate(need)}
            zp = lat[[idx[r] for r in path]]
            e_move = (zp - lat[idx[path[-1]]]).abs().mean(dim=(1, 2)).numpy()
            b_move = barrier(e_move)
            for k in range(len(path) - 1):
                if pst[k] in range(1, 10):
                    stats["move_end"][int(pst[k])].append((e_move[k + 1] > e_move[k], b_move[k]))
            for k in range(len(path) - 1):  # stage-end subgoal: energy of the next path frame vs this one, same goal
                if pst[k] not in range(1, 10) or path[k + 1] > stage_end[path[k]]:
                    continue
                g = lat[idx[stage_end[path[k]]]]
                e0 = float((zp[k] - g).abs().mean())
                e1 = float((zp[k + 1] - g).abs().mean())
                stats["stage_end"][int(pst[k])].append((e1 > e0, np.nan))
            ep_rows += path
            ep_stage += pst.tolist()
        if kinds[sel[0]] == 3 and len(real_moves) == 15:  # whole expert episode: the 15-move goal
            lat = encode([h["pixels"][r] for r in ep_rows])
            e_ep = (lat - lat[-1]).abs().mean(dim=(1, 2)).numpy()
            b_ep = barrier(e_ep)
            for k in range(len(ep_rows) - 1):
                if ep_stage[k] in range(1, 10):
                    stats["episode_end_15_moves"][int(ep_stage[k])].append((e_ep[k + 1] > e_ep[k], b_ep[k]))
            episode_profiles.append({"file": int(key[0]), "segment": int(key[1]), "energy": e_ep.tolist()})
        print(f"segment {tuple(int(x) for x in key)}: {len(real_moves)} moves", flush=True)

    def summarize(items):
        up = np.array([u for u, _ in items], bool)
        b = np.array([x for _, x in items], float)
        fin = b[np.isfinite(b)]
        out = {"steps": int(len(up)), "uphill_fraction": float(up.mean()) if len(up) else float("nan")}
        if np.isfinite(b).any() or np.isinf(b).any():
            if not np.all(np.isnan(b)):
                out |= {
                    "barrier_steps_median": float(np.median(fin)) if len(fin) else float("inf"),
                    "barrier_steps_p90": float(np.percentile(fin, 90)) if len(fin) else float("inf"),
                    f"barrier_beyond_rollout_{args.rollout}": float(np.mean(b > args.rollout)),
                }
        return out

    result = {"archive": args.archive, "checkpoint": args.checkpoint, "model_step_s": FSTP / 30.0}
    for goal, by_stage in stats.items():
        allitems = [x for v in by_stage.values() for x in v]
        result[goal] = {
            "overall": summarize(allitems),
            "by_stage": {STAGES[s]: summarize(v) for s, v in sorted(by_stage.items())},
        }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    if episode_profiles:
        with open(args.out.replace(".json", "_episodes.json"), "w") as f:
            json.dump(episode_profiles, f)
    for goal in stats:
        o = result[goal]["overall"]
        print(f"[{goal}] {o['steps']} expert steps: uphill {o['uphill_fraction']:.2f}"
              + (f", barrier median {o.get('barrier_steps_median')} steps" if "barrier_steps_median" in o else ""),
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()

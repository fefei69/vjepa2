# Action recovery by planning: can the world model find the action the robot actually took?
#
#   python -m app.vjepa_hanoi.plan_eval --fname <yaml> --checkpoint <ckpt> --split test --horizon 1 --max_windows 1024
#
# For each held-out clip: encode frame 0 (context, with its measured state) and frame H (the goal, H model steps =
# H x 0.267 s later), then run the repo's planner -- notebooks/utils/mpc_utils.py::cem with the step function of
# notebooks/utils/world_model_wrapper.py and its default settings -- to find H actions [dx, dy, dz, dgrip] (rotation
# fixed at 0) whose imagined rollout ends closest (mean L1) to the goal latent. The plan is compared with the actions
# actually taken (differences of measured states, the training labels):
#   xyz error (mm) of the summed H-step displacement against a zero-action baseline (for H=1 this is the action);
#   per-axis Pearson r and direction cosine on moving clips; gripper sign accuracy on open/close events (net change);
#   energy ranking: is the energy of the true actions below that of doing nothing (tests the model, not the search).
# For H > 1 the planner scores only the final frame, so how motion is split across steps is not identified: the
# headline uses the summed displacement; first-action numbers are kept for reference only.
# --goal predicted replaces the real goal with the model's own rollout under the true actions (gripper clipped to the
# planner's +-0.75 so the goal is reachable): a planner calibration whose action error is search error plus any
# flatness of the model's energy; its energy-ranking numbers are trivial by construction and are not reported.
# --from_clips re-computes the summary from a saved *_clips.npz without planning again.

import argparse
import importlib.util
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from app.vjepa_droid.transforms import make_transforms
from app.vjepa_hanoi.ac_step import encode_clips
from app.vjepa_hanoi.eval import STAGES, build_models, strip_prefixes
from app.vjepa_hanoi.hanoi import HanoiClipDataset

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_mpc_utils():
    """notebooks/utils is not a package; load the planner module by path."""
    spec = importlib.util.spec_from_file_location(
        "mpc_utils", os.path.join(REPO, "notebooks", "utils", "mpc_utils.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    logging.getLogger("mpc_utils").setLevel(logging.WARNING)  # cem logs every iteration
    return mod


def make_step_fn(predictor, tokens_per_frame, normalize_reps, compute_new_pose):
    """Same as notebooks/utils/world_model_wrapper.py::WorldModel.infer_next_action.step_predictor."""

    def step_predictor(reps, actions, poses):
        B, T, N_T, D = reps.size()
        reps = reps.flatten(1, 2)
        next_rep = predictor(reps, actions, poses)[:, -tokens_per_frame:]
        if normalize_reps:
            next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
        next_rep = next_rep.view(B, 1, N_T, D)
        next_pose = compute_new_pose(poses[:, -1:], actions[:, -1:])
        return next_rep, next_pose

    return step_predictor


def rollout(step_fn, frame, pose, actions):
    """frame [N,1,HW,D], pose [N,1,7], actions [N,H,7] -> final imagined frame [N,1,HW,D] (as cem rolls out)."""
    frames, poses = frame, pose
    for h in range(actions.size(1)):
        nxt, npose = step_fn(frames, actions[:, : h + 1], poses)
        frames, poses = torch.cat([frames, nxt], dim=1), torch.cat([poses, npose], dim=1)
    return frames[:, -1:]


def pearson(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize(rec, moving_mm, goal="real"):
    t1, p1 = rec["true_first"][:, :3] * 1e3, rec["plan_first"][:, :3] * 1e3
    ts, ps = rec["true_sum"][:, :3] * 1e3, rec["plan_sum"][:, :3] * 1e3
    mag1, err1 = np.linalg.norm(t1, axis=1), np.linalg.norm(p1 - t1, axis=1)
    mags, errs = np.linalg.norm(ts, axis=1), np.linalg.norm(ps - ts, axis=1)
    mov1, movs = mag1 >= moving_mm, mags >= moving_mm  # moving by the first action / by the net H-step displacement
    out = {
        "clips": int(len(t1)),
        "moving_clips": int(movs.sum()),
        "moving_clips_first_action": int(mov1.sum()),
        "moving_threshold_mm": moving_mm,
    }

    def block(err, mag, true, plan, m):
        cos = np.sum(true[m] * plan[m], axis=1) / (
            np.linalg.norm(true[m], axis=1) * np.linalg.norm(plan[m], axis=1) + 1e-9
        )
        return {
            "err_mm_mean": float(err.mean()),
            "err_mm_median": float(np.median(err)),
            "zero_action_err_mm_mean": float(mag.mean()),
            "moving_err_mm_median": float(np.median(err[m])) if m.any() else float("nan"),
            "moving_zero_action_err_mm_median": float(np.median(mag[m])) if m.any() else float("nan"),
            "moving_err_over_zero_median": float(np.median(err[m] / mag[m])) if m.any() else float("nan"),
            "moving_direction_cosine_median": float(np.median(cos)) if m.any() else float("nan"),
            "moving_pearson_r_xyz": [pearson(true[m, i], plan[m, i]) for i in range(3)],
            # on static steps the error is the planner's spurious motion (cem's final mean never reaches exactly 0)
            "static_err_mm_median": float(np.median(err[~m])) if (~m).any() else float("nan"),
        }

    out["summed_displacement"] = block(errs, mags, ts, ps, movs)  # headline (identical to first_action when H=1)
    out["first_action"] = block(err1, mag1, t1, p1, mov1)  # not identified for H > 1: reference only
    tg, pg = rec["true_sum"][:, 6], rec["plan_sum"][:, 6]  # net gripper change over the horizon
    ev = np.abs(tg) >= 0.25
    out["gripper"] = {
        "event_clips": int(ev.sum()),
        "event_sign_accuracy": float(np.mean(np.sign(pg[ev]) == np.sign(tg[ev]))) if ev.any() else float("nan"),
        "no_event_clips": int((~ev).sum()),
        "no_event_kept_zero": float(np.mean(pg[~ev] == 0)) if (~ev).any() else float("nan"),
    }
    e_true, e_zero, e_plan = rec["energy_true"], rec["energy_zero"], rec["energy_plan"]
    out["energy"] = {
        "mean_true": float(e_true.mean()),
        "mean_zero": float(e_zero.mean()),
        "mean_plan": float(e_plan.mean()),
    }
    if goal == "real":  # with a predicted goal E(true) ~ 0 by construction, so rankings say nothing about the model
        out["energy"] |= {
            "true_below_zero_all": float(np.mean(e_true < e_zero)),
            "true_below_zero_moving": float(np.mean(e_true[movs] < e_zero[movs])) if movs.any() else float("nan"),
            "plan_below_true": float(np.mean(e_plan < e_true)),
        }
    else:
        out["energy"]["search_gap_mean"] = float((e_plan - e_true).mean())  # energy the search leaves on the table
    return out


def headline(result):
    o = result["overall"]
    s, gr, en = o["summed_displacement"], o["gripper"], o["energy"]
    r_xyz = [round(r, 2) for r in s["moving_pearson_r_xyz"]]
    msg = (
        f"[{result['split']} H={result['horizon_steps']} goal={result['goal']}] {o['clips']} clips "
        f"({o['moving_clips']} moving) | net xyz err on moving: median {s['moving_err_mm_median']:.2f} mm vs "
        f"zero-action {s['moving_zero_action_err_mm_median']:.2f} (ratio {s['moving_err_over_zero_median']:.2f}), cos "
        f"{s['moving_direction_cosine_median']:.2f}, r_xyz {r_xyz} | static: spurious "
        f"{s['static_err_mm_median']:.2f} mm | gripper events {gr['event_sign_accuracy']:.2f} ({gr['event_clips']})"
    )
    if result["goal"] == "real":
        msg += f" | E(true)<E(zero) on moving {en['true_below_zero_moving']:.2f}"
    else:
        msg += f" | search energy gap {en['search_gap_mean']:.4f}"
    return msg


def finalize(rec, meta, moving_mm, out):
    rec = {k: np.asarray(v) for k, v in rec.items()}
    result = dict(meta)
    result["overall"] = summarize(rec, moving_mm, meta["goal"])
    result["by_stage"] = {
        STAGES.get(int(st), str(st)): summarize(
            {k: v[rec["stage"] == st] for k, v in rec.items()}, moving_mm, meta["goal"]
        )
        for st in np.unique(rec["stage"])
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(headline(result) + f" -> {out}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fname", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--from_clips", default=None, help="re-summarize a saved *_clips.npz (needs --horizon, --goal)")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--horizon", type=int, default=1, help="goal = frame H of the clip; plan H actions")
    p.add_argument("--goal", choices=("real", "predicted"), default="real")
    p.add_argument("--max_windows", type=int, default=1024)
    # repo planner defaults (notebooks/utils/world_model_wrapper.py)
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--cem_steps", type=int, default=10)
    p.add_argument("--momentum_mean", type=float, default=0.15)
    p.add_argument("--momentum_std", type=float, default=0.15)
    p.add_argument("--maxnorm", type=float, default=0.05)
    p.add_argument("--moving_mm", type=float, default=5.0, help="moving step: |true dxyz| >= this (mm per step)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.from_clips:
        z = np.load(args.from_clips)
        rec = {k: z[k] for k in z.files}
        old = args.from_clips.replace("_clips.npz", ".json")
        meta = json.load(open(old)) if os.path.exists(old) else {}
        meta = {k: v for k, v in meta.items() if k not in ("overall", "by_stage")}
        meta |= {"horizon_steps": args.horizon, "goal": args.goal, "split": meta.get("split", args.split)}
        finalize(rec, meta, args.moving_mm, args.out or old)
        return
    assert args.fname and args.checkpoint, "--fname and --checkpoint are required unless --from_clips"

    with open(args.fname) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    d, loss_cfg = cfg["data"], cfg["loss"]
    crop, fpc = d.get("crop_size", 256), max(d["dataset_fpcs"])
    assert 1 <= args.horizon < fpc
    tokens_per_frame = (crop // d["patch_size"]) ** 2
    normalize_reps = loss_cfg["normalize_reps"]
    dtype = torch.bfloat16 if str(cfg["meta"].get("dtype", "float32")).lower() == "bfloat16" else torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder, predictor = build_models(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    print("encoder <- target_encoder:", encoder.load_state_dict(strip_prefixes(ckpt["target_encoder"]), strict=True))
    print("predictor <- predictor:", predictor.load_state_dict(strip_prefixes(ckpt["predictor"]), strict=True))
    epoch = ckpt.get("epoch")
    del ckpt
    encoder.eval()
    predictor.eval()

    mpc = load_mpc_utils()
    step_fn = make_step_fn(predictor, tokens_per_frame, normalize_reps, mpc.compute_new_pose)
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop,
    )
    dataset = HanoiClipDataset(
        d["val_dataset" if args.split == "val" else "test_dataset"],
        frames_per_clip=fpc,
        fps=d["fps"],
        transform=transform,
        max_windows=args.max_windows,
    )
    H = args.horizon
    cem_kw = dict(
        rollout=H,
        samples=args.samples,
        topk=args.topk,
        cem_steps=args.cem_steps,
        momentum_mean=args.momentum_mean,
        momentum_std=args.momentum_std,
        maxnorm=args.maxnorm,
    )

    keys = ("true_first", "plan_first", "true_sum", "plan_sum", "energy_true", "energy_zero", "energy_plan", "stage")
    rec = {k: [] for k in keys}
    t0 = time.time()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
        for i in range(len(dataset)):
            buffer, actions, states, _, _, _ = dataset[i]
            clip = buffer[None].to(device)
            z = encode_clips(encoder, clip[:, :, [0, H]], normalize_reps).float()
            z0 = z[:, :tokens_per_frame].view(1, 1, tokens_per_frame, -1)
            pose = torch.as_tensor(states[None, :1], dtype=torch.float32, device=device)
            true = torch.as_tensor(actions[None, :H], dtype=torch.float32, device=device)
            # the planner's action space: rotation fixed at 0 (true rotation deltas are < 0.01 rad)
            true[..., 3:6] = 0
            if args.goal == "real":
                zg = z[:, tokens_per_frame:].view(1, 1, tokens_per_frame, -1)
            else:
                true[..., 6].clamp_(-0.75, 0.75)  # reachable by cem, which clips gripper samples to +-0.75
                zg = rollout(step_fn, z0, pose, true).float()

            torch.manual_seed(args.seed + i)
            plan = mpc.cem(context_frame=z0, context_pose=pose, goal_frame=zg, world_model=step_fn, **cem_kw)[0]

            cand = torch.stack([true[0], torch.zeros_like(true[0]), plan.float()])  # [3, H, 7]
            final = rollout(step_fn, z0.repeat(3, 1, 1, 1), pose.repeat(3, 1, 1), cand).float()
            energy = torch.mean(torch.abs(final.flatten(1) - zg.flatten(1)), dim=-1).cpu().numpy()

            tn, pn = true[0].cpu().numpy(), plan.float().cpu().numpy()
            rec["true_first"].append(tn[0])
            rec["plan_first"].append(pn[0])
            rec["true_sum"].append(tn.sum(0))
            rec["plan_sum"].append(pn.sum(0))
            rec["energy_true"].append(energy[0])
            rec["energy_zero"].append(energy[1])
            rec["energy_plan"].append(energy[2])
            rec["stage"].append(int(dataset.win_stage[i]))
            if (i + 1) % 50 == 0 or i == 0:
                print(f"{i + 1}/{len(dataset)} clips, {(time.time() - t0) / (i + 1):.2f} s/clip", flush=True)

    meta = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "epoch": epoch,
        "split": args.split,
        "horizon_steps": H,
        "horizon_s": H * dataset.fstp / dataset.rate_hz,
        "goal": args.goal,
        "planner": cem_kw | {"source": "notebooks/utils/mpc_utils.py::cem"},
        "seconds_per_clip": (time.time() - t0) / len(dataset),
    }
    out = args.out or f"{args.checkpoint}.{args.split}_plan_h{H}_{args.goal}.json"
    np.savez_compressed(out.replace(".json", "_clips.npz"), **{k: np.asarray(v) for k, v in rec.items()})
    finalize(rec, meta, args.moving_mm, out)


if __name__ == "__main__":
    main()

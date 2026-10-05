# Held-out evaluation of a V-JEPA 2-AC checkpoint on the play_k5 validation or test walks (single process).
#
#   python -m app.vjepa_hanoi.eval --fname <yaml> --checkpoint <folder>/best.pt --split test
#
# Per clip (every `window_stride`-th valid clip of the split):
#   tf          teacher-forced one-step L1, frames 1..T-1 from ground-truth context (the training jloss)
#   rollout     auto-regressive L1 over frames 1..auto_steps (the training sloss)
#   copy_tf / copy_rollout   the same targets predicted by repeating the last given frame: a no-dynamics floor
#   shuffled_tf teacher-forced L1 with actions taken from another, unrelated clip of the split (fixed-seed shuffled
#               batches, actions rolled by one): the gap to `tf` measures how much the predictor uses its actions
# Reported overall, by motion stage of the clip's first frame and by segment kind; written as JSON.

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import yaml

from app.vjepa_droid.transforms import make_transforms
from app.vjepa_droid.utils import init_video_model
from app.vjepa_hanoi.ac_step import encode_clips, per_clip_loss, predict
from app.vjepa_hanoi.hanoi import init_eval_data

STAGES = {
    0: "unknown",
    1: "open",
    2: "approach_source",
    3: "descend_source",
    4: "grasp",
    5: "lift",
    6: "transit",
    7: "insert_target",
    8: "release",
    9: "retreat",
    10: "return_initial_xy",
    11: "return_initial_z",
    12: "hold",
}
KINDS = {0: "walk", 1: "crop", 2: "clip"}
METRICS = ("tf", "rollout", "copy_tf", "copy_rollout", "shuffled_tf")


def strip_prefixes(state_dict):
    out = {}
    for k, v in state_dict.items():
        for p in ("module.", "backbone."):
            if k.startswith(p):
                k = k[len(p) :]
        out[k] = v
    return out


def build_models(cfg, device):
    m, d, meta = cfg["model"], cfg["data"], cfg["meta"]
    return init_video_model(
        device=device,
        patch_size=d["patch_size"],
        max_num_frames=m.get("max_num_frames", 64),
        tubelet_size=d["tubelet_size"],
        model_name=m["model_name"],
        crop_size=d.get("crop_size", 256),
        pred_depth=m["pred_depth"],
        pred_num_heads=m.get("pred_num_heads", None),
        pred_embed_dim=m["pred_embed_dim"],
        action_embed_dim=m.get("action_embed_dim", 7),
        pred_is_frame_causal=m.get("pred_is_frame_causal", True),
        use_extrinsics=m.get("use_extrinsics", False),
        use_sdpa=meta.get("use_sdpa", False),
        use_silu=m.get("use_silu", False),
        use_pred_silu=m.get("use_pred_silu", False),
        wide_silu=m.get("wide_silu", True),
        use_rope=m.get("use_rope", False),
        uniform_power=m.get("uniform_power", False),
        use_activation_checkpointing=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True, help="training config")
    parser.add_argument("--checkpoint", required=True, help="best.pt / latest.pt written by app.vjepa_hanoi.train")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--window_stride", type=int, default=3, help="every k-th valid clip (default 3)")
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out", default=None, help="JSON path (default: <checkpoint>.<split>_eval.json)")
    args = parser.parse_args()

    with open(args.fname) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    d, loss_cfg = cfg["data"], cfg["loss"]
    archive = d["val_dataset" if args.split == "val" else "test_dataset"]
    crop_size, patch_size = d.get("crop_size", 256), d["patch_size"]
    tokens_per_frame = (crop_size // patch_size) ** 2
    fpc = max(d["dataset_fpcs"])
    auto_steps = min(loss_cfg.get("auto_steps", 1), fpc)
    normalize_reps, loss_exp = loss_cfg["normalize_reps"], loss_cfg["loss_exp"]
    dtype = torch.bfloat16 if str(cfg["meta"].get("dtype", "float32")).lower() == "bfloat16" else torch.float32

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder, predictor = build_models(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    print("encoder <- target_encoder:", encoder.load_state_dict(strip_prefixes(ckpt["target_encoder"]), strict=True))
    print("predictor <- predictor:", predictor.load_state_dict(strip_prefixes(ckpt["predictor"]), strict=True))
    encoder.eval()
    predictor.eval()
    epoch = ckpt.get("epoch")
    del ckpt

    # square frames: a deterministic full-frame resize, identical to the training aug's fallback path
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )
    loader, dataset = init_eval_data(
        data_path=archive,
        batch_size=args.batch_size,
        frames_per_clip=fpc,
        fps=d["fps"],
        transform=transform,
        window_stride=args.window_stride,
        max_windows=args.max_windows,
        num_workers=args.num_workers,
        pin_mem=device.type == "cuda",
        collator=torch.utils.data.default_collate,
        shuffle_seed=0,  # batch neighbours are unrelated clips, so rolled actions are a real mismatch
    )

    per_clip = defaultdict(list)
    index = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
        for b in loader:
            c = b[0].to(device, non_blocking=True)
            a = b[1].to(device, dtype=torch.float)
            s = b[2].to(device, dtype=torch.float)
            e = b[3].to(device, dtype=torch.float)
            h = encode_clips(encoder, c, normalize_reps)
            z_tf, z_ar = predict(predictor, h, a, s, e, tokens_per_frame, auto_steps, normalize_reps)
            last_ctx = h[:, :-tokens_per_frame]  # copy baseline: z_hat_{t+1} = z_t
            first = h[:, :tokens_per_frame].repeat(1, auto_steps, 1)  # rollout copy baseline: z_hat = z_0
            vals = {
                "tf": per_clip_loss(z_tf.float(), h.float(), tokens_per_frame, loss_exp),
                "rollout": per_clip_loss(z_ar.float(), h.float(), tokens_per_frame, loss_exp),
                "copy_tf": per_clip_loss(last_ctx.float(), h.float(), tokens_per_frame, loss_exp),
                "copy_rollout": per_clip_loss(first.float(), h.float(), tokens_per_frame, loss_exp),
            }
            if c.size(0) > 1:
                z_sh, _ = predict(predictor, h, a.roll(1, dims=0), s, e, tokens_per_frame, 1, normalize_reps)
                vals["shuffled_tf"] = per_clip_loss(z_sh.float(), h.float(), tokens_per_frame, loss_exp)
            else:
                vals["shuffled_tf"] = torch.full((1,), float("nan"), device=device)
            for k, v in vals.items():
                per_clip[k].append(v.cpu().numpy())
            index.append(b[5].numpy())

    index = np.concatenate(index)
    per_clip = {k: np.concatenate(v) for k, v in per_clip.items()}

    def summary(mask):
        return {k: float(np.nanmean(per_clip[k][mask])) for k in METRICS} | {"clips": int(mask.sum())}

    stage, kind = dataset.win_stage[index], dataset.win_kind[index]
    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "epoch": epoch,
        "split": args.split,
        "archive": archive,
        "window_stride": args.window_stride,
        "overall": summary(np.ones(len(index), dtype=bool)),
        "by_stage": {STAGES.get(int(st), str(st)): summary(stage == st) for st in np.unique(stage)},
        "by_kind": {KINDS[int(k)]: summary(kind == k) for k in np.unique(kind)},
    }
    out = args.out or f"{args.checkpoint}.{args.split}_eval.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    o = result["overall"]
    print(
        f"[{args.split}] {o['clips']} clips | tf {o['tf']:.4f} (copy {o['copy_tf']:.4f}, shuffled actions "
        f"{o['shuffled_tf']:.4f}) | rollout {o['rollout']:.4f} (copy {o['copy_rollout']:.4f}) -> {out}"
    )


if __name__ == "__main__":
    main()

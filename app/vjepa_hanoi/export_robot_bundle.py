# Export what the planning server needs to run on the robot machine, without the cluster's checkpoints or recordings.
#
#   python -m app.vjepa_hanoi.export_robot_bundle \
#       --fname configs/train/vitg16/hanoi-firstmove-ac-ft-256px-8f.yaml \
#       --checkpoint /scratch/cw5167/checkpoints/vjepa2_ac_hanoi/A_firstmove_vitg_finetune_v1/best.pt \
#       --goal_archive /scratch/cw5167/checkpoints/vjepa2_ac_hanoi/data/expert6_first_move/train.npz \
#       --out /scratch/cw5167/checkpoints/vjepa2_ac_hanoi/robot_bundle_firstmove
#
# Writes into --out:
#   weights.pt    the frozen encoder ('target_encoder') and the predictor, float32 as trained; no optimizer state
#   goals.npz     per board transition of the goal archive: the three paper-protocol goal frames (224 px, from the
#                 first complete training demo of that move, as goal_planner picks them), that move's schedule, the
#                 global schedule, and the training data's workspace box
#   config.yaml   the training config (the server reads the model shape from it)
#   MANIFEST.json sources, sha256 of every file, and the command to start the server
# Then, on the robot machine:
#   python -m app.vjepa_hanoi.goal_planner --fname <out>/config.yaml --checkpoint <out>/weights.pt \
#       --goal_bundle <out>/goals.npz --protocol paper --host 127.0.0.1 --port 8766

import argparse
import hashlib
import json
import os
import shutil

import h5py
import numpy as np
import torch

from app.vjepa_hanoi.paper_protocol_eval import goal_rows, moves, time_schedule


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fname", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--goal_archive", required=True)
    p.add_argument("--final_stage", type=int, choices=(8, 9), default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    keep = {k: ckpt[k] for k in ("target_encoder", "predictor")}
    info = {k: ckpt.get(k) for k in ("epoch", "val_loss", "loss") if k in ckpt}
    torch.save(keep | {"source_checkpoint": os.path.abspath(args.checkpoint)} | info,
               os.path.join(args.out, "weights.pt"))  # fmt: skip
    del ckpt, keep

    train, paths, _ = moves(args.goal_archive)
    transitions, frames, sources, schedules = [], [], [], []
    for m in train:  # the first complete training demo of each transition, as goal_planner.set_goal picks it
        if m["transition"] in transitions:
            continue
        g = goal_rows(m, args.final_stage)
        with h5py.File(paths[m["file"]], "r") as h:
            frames.append(np.stack([h["pixels"][g[k]] for k in (1, 2, 3)]))
        reach = time_schedule([x for x in train if x["transition"] == m["transition"]], args.final_stage)
        transitions.append(m["transition"])
        sources.append([[m["file"], g[k]] for k in (1, 2, 3)])
        schedules.append([reach[0], reach[1] - reach[0], reach[2] - reach[1]])
    reach = time_schedule(train, args.final_stage)
    z = np.load(args.goal_archive)
    xyz = z["measured_xyz"]
    np.savez_compressed(
        os.path.join(args.out, "goals.npz"),
        transitions=np.array(transitions),
        frames=np.stack(frames).astype(np.uint8),
        frame_sources=np.array(sources, np.int64),
        source_paths=np.array(paths),
        schedules=np.array(schedules, np.int64),
        global_schedule=np.array([reach[0], reach[1] - reach[0], reach[2] - reach[1]], np.int64),
        box_min=xyz.min(0) - np.array([0.005, 0.005, 0.001]),
        box_max=xyz.max(0) + 0.005,
        final_stage=np.array(args.final_stage),
        goal_archive=np.array(os.path.abspath(args.goal_archive)),
    )
    shutil.copyfile(args.fname, os.path.join(args.out, "config.yaml"))
    files = {f: sha256(os.path.join(args.out, f)) for f in ("weights.pt", "goals.npz", "config.yaml")}
    with open(os.path.join(args.out, "MANIFEST.json"), "w") as f:
        json.dump({"source_checkpoint": os.path.abspath(args.checkpoint), "checkpoint_info": info,
                   "config": os.path.abspath(args.fname), "goal_archive": os.path.abspath(args.goal_archive),
                   "final_stage": args.final_stage, "transitions": dict(zip(transitions, schedules)),
                   "sha256": files,
                   "serve": "python -m app.vjepa_hanoi.goal_planner --fname <bundle>/config.yaml --checkpoint "
                            "<bundle>/weights.pt --goal_bundle <bundle>/goals.npz --protocol paper "
                            "--host 127.0.0.1 --port 8766"}, f, indent=1)  # fmt: skip
    print(f"{len(transitions)} transitions {transitions}; schedules {schedules}; -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

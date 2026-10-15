# Per-axis range of the actions a model was trained on, for the planner option --sampling data.
#
#   python -m app.vjepa_hanoi.action_bounds --archive <train.npz> --fps 1 [--out bounds.json]
#
# CEM samples [dx, dy, dz, gripper] from a Gaussian and clips it to a box. With the repo box (0.05 m per fps-4 step,
# scaled with the step: 0.1875 m at fps 1) almost every first-iteration sample is a larger step than any in training
# (expert6_first_move: largest per-axis step 0.026 m at fps 4, 0.083 m at fps 1), so CEM ranks candidates the world model
# never saw. These bounds come from the training split's own actions, every action of every training clip (each counted
# as often as training sees it), computed exactly as the loader does (hanoi.py: states every fstp rows, poses_to_diffs).
# They describe the data's action range only: no state, stage or episode information goes into the planner.
import argparse
import json

import h5py
import numpy as np

from app.vjepa_hanoi.hanoi import HanoiClipDataset, hanoi_states, poses_to_diffs


def training_actions(archive, fps, frames_per_clip=8):
    """[N, 7] actions of every training clip of the archive at this fps (window-weighted, as the training loader)."""
    ds = HanoiClipDataset(archive, frames_per_clip=frames_per_clip, fps=fps)  # window index only, no pixels read
    offsets = np.arange(frames_per_clip) * ds.fstp
    out = []
    for f in np.unique(ds.win_file):
        with h5py.File(ds.files[f], "r") as h:
            state, proprio = h["state"][:], h["proprio"][:]
        for start in ds.win_start[ds.win_file == f]:
            rows = start + offsets
            out.append(poses_to_diffs(hanoi_states(state[rows], proprio[rows])))
    return np.concatenate(out), ds.fstp, len(ds)


def action_bounds(archive, fps, pct=0.1, moving_mm=5.0):
    """Signed per-axis xyz box [pct, 100 - pct] percentiles and the initial CEM std per axis (spread of the moving
    steps, |xyz| >= moving_mm), from the training actions at this fps."""
    a, fstp, clips = training_actions(archive, fps)
    xyz = a[:, :3]
    moving = np.linalg.norm(xyz, axis=1) >= moving_mm / 1e3
    return {
        "lo": np.percentile(xyz, pct, axis=0).tolist(),
        "hi": np.percentile(xyz, 100 - pct, axis=0).tolist(),
        "std": xyz[moving].std(axis=0).tolist(),
        "max_abs": np.abs(xyz).max(axis=0).tolist(),
        "max_norm": float(np.linalg.norm(xyz, axis=1).max()),
        "gripper_nonzero": float((np.abs(a[:, 6]) > 1e-6).mean()),
        "archive": archive,
        "fps": fps,
        "step_rows": fstp,
        "percentile": pct,
        "moving_mm": moving_mm,
        "actions": int(len(a)),
        "clips": clips,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True, help="training split .npz")
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    b = action_bounds(args.archive, args.fps)
    print(json.dumps(b, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(b, f, indent=1)


if __name__ == "__main__":
    main()

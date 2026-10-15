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


def kmeans(x, k, seed=0, iters=100):
    """Plain k-means (k-means++ start), deterministic for a seed. x [N, d] -> centers [k, d], labels [N]."""
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d2 = np.min(((x[:, None] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(x[rng.choice(len(x), p=d2 / d2.sum())])
    centers = np.asarray(centers)
    for _ in range(iters):
        labels = np.argmin(((x[:, None] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.stack([x[labels == j].mean(0) if (labels == j).any() else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return centers, labels


def action_codebook(archive, fps, k_move=48, moving_mm=5.0, grip=0.5, seed=0):
    """A finite action set for the categorical planner (--sampling cat), from the training actions at this fps.
    Codes are rows [dx, dy, dz, gripper]: 'stay' (0, 0, 0, 0), 'close' (0, 0, 0, +1), 'open' (0, 0, 0, -1) (the expert
    opens and closes only while still: |xyz| <= 4.4 mm at fps 1), and the k_move k-means centers of the moving steps
    (|xyz| >= moving_mm, gripper unchanged). prior: each code's share of the training actions (nearest code)."""
    a, fstp, clips = training_actions(archive, fps)
    xyz, g = a[:, :3], a[:, 6]
    is_grip = np.abs(g) >= grip
    moving = ~is_grip & (np.linalg.norm(xyz, axis=1) >= moving_mm / 1e3)
    centers, labels = kmeans(xyz[moving], k_move, seed)
    codes = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]]
    codes += [[float(c[0]), float(c[1]), float(c[2]), 0.0] for c in centers]
    prior = [float((~is_grip & ~moving).mean()), float((g >= grip).mean()), float((g <= -grip).mean())]
    prior += [float((labels == j).sum() / len(a)) for j in range(k_move)]
    return {"codes": codes, "prior": prior, "names": ["stay", "close", "open"] + [f"move{j}" for j in range(k_move)],
            "fps": fps, "step_rows": fstp, "k_move": k_move, "moving_mm": moving_mm, "archive": archive,
            "actions": int(len(a)), "clips": clips}


def limit_codebook(cb, max_step_m):
    """The codebook without the moving codes longer than max_step_m (|xyz|, metres); stay/close/open are kept. For a
    planner whose executed steps are capped (goal_planner --max_speed): it then only plans steps it will execute."""
    keep = [i for i, c in enumerate(cb["codes"]) if np.linalg.norm(c[:3]) <= max_step_m + 1e-9]
    out = dict(cb)
    for k in ("codes", "prior", "names"):
        out[k] = [cb[k][i] for i in keep]
    out["max_step_m"] = float(max_step_m)
    return out


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

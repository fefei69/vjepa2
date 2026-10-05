# Tower-of-Hanoi play_k5 data -> V-JEPA 2-AC training clips.
#
# The split archives (openpi data/hanoi/play_k5_pi05/indices/play_{train,val,test}.npz, cross-checked identical to the
# Cosmos play_k5 archive) decide only WHICH raw rows are usable observations (handover section 3) and which raw spans
# form a split's segments (section 2). Frames and robot state are read from the raw recordings the archive names in
# `source_paths`; the archive's policy labels, goals and joint-space states are not used by a world model.
#
# A clip is `frames_per_clip` usable rows spaced fstp = ceil(rate_hz / fps) raw rows apart inside one segment
# (30 Hz, fps 4 -> every 8th row, 3.75 Hz: the DROID sampling rate). Each item follows the contract of
# app/vjepa_droid/droid.py, which app/vjepa_droid/train.py consumes:
#     (buffer [C, T, H, W] after transform, actions [T-1, 7], states [T, 7], extrinsics [T, 6] zeros, rows [T], index)
# state  = [measured xyz (m), Euler 'xyz' of the measured angle-axis (rad), gripper closedness (0 open, 1 closed)]
# action = difference of consecutive states, computed exactly as droid.py::poses_to_diffs.

from logging import getLogger
from math import ceil

import h5py
import numpy as np
import torch
import torch.utils.data
from scipy.spatial.transform import Rotation

logger = getLogger()

# Measured jaw stroke proprio[:, 6] at a grasp depends on the held ring (medians 0.0153 / 0.0203 / 0.0257 / 0.0307 m
# for rings 1-4; open reads 0.0340). These thresholds are the open p1 and the ring-4-held p99, so every grasp reads 1.0
# and closedness transitions stay within one model step of the physical 50%-travel point (measured over 5,455 events).
GRIPPER_OPEN = 0.0340
GRIPPER_CLOSED = 0.0309


def poses_to_diffs(poses):
    """Same computation as app/vjepa_droid/droid.py::DROIDVideoDataset.poses_to_diffs."""
    xyz = poses[:, :3]
    thetas = poses[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    xyz_diff = xyz[1:] - xyz[:-1]
    angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    angle_diff = np.stack([d for d in angle_diff], axis=0)
    closedness = poses[:, -1:]
    closedness_delta = closedness[1:] - closedness[:-1]
    return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)


def hanoi_states(state, proprio, gripper_open=GRIPPER_OPEN, gripper_closed=GRIPPER_CLOSED):
    """Raw `state` (xyz + angle-axis) and `proprio` rows -> [T, 7] AC states."""
    state = state.astype(np.float64)
    euler = Rotation.from_rotvec(state[:, 3:6]).as_euler("xyz", degrees=False)
    jaw = proprio[:, 6].astype(np.float64)
    closedness = np.clip((gripper_open - jaw) / (gripper_open - gripper_closed), 0.0, 1.0)
    return np.concatenate([state[:, :3], euler, closedness[:, None]], axis=1)


class HanoiClipDataset(torch.utils.data.Dataset):
    """Every valid clip of one split; item i is a fixed window, so a shuffling sampler covers windows uniformly."""

    def __init__(
        self,
        archive_path,
        frames_per_clip=8,
        fps=4,
        transform=None,
        window_stride=1,
        max_windows=None,
    ):
        self.archive_path = archive_path
        self.frames_per_clip = frames_per_clip
        self.transform = transform

        z = np.load(archive_path)
        self.split = str(z["split"])
        self.files = [str(p) for p in z["source_paths"]]
        file_idx = z["file_indices"]
        obs = z["source_observation_indices"]
        bounds = z["segment_bounds"]
        kinds = z["segment_kinds"]
        stages = z["motion_stages"]

        rates = set()
        for fn in self.files:
            with h5py.File(fn, "r") as f:
                rates.add(float(f.attrs["rate_hz"]))
        assert len(rates) == 1, f"recordings disagree on rate_hz: {rates}"
        self.rate_hz = rates.pop()
        self.fstp = ceil(self.rate_hz / fps)
        span = (frames_per_clip - 1) * self.fstp + 1

        # usable rows of each segment -> starts whose frames_per_clip sampled rows are all usable observations
        seg_keys = np.unique(np.stack([file_idx, bounds[:, 0], bounds[:, 1], kinds], axis=1), axis=0)
        win_file, win_start, win_kind, win_stage = [], [], [], []
        for f, lo, hi, kind in seg_keys:
            sel = (file_idx == f) & (bounds[:, 0] == lo) & (bounds[:, 1] == hi)
            usable = np.zeros(hi - lo, dtype=bool)
            usable[obs[sel] - lo] = True
            stage_of = np.zeros(hi - lo, dtype=np.int64)
            stage_of[obs[sel] - lo] = stages[sel]
            n = len(usable) - span + 1
            if n <= 0:
                continue
            ok = np.ones(n, dtype=bool)
            for j in range(frames_per_clip):
                ok &= usable[j * self.fstp : j * self.fstp + n]
            starts = np.nonzero(ok)[0]
            win_file.append(np.full(len(starts), f, dtype=np.int64))
            win_start.append(lo + starts)
            win_kind.append(np.full(len(starts), kind, dtype=np.int64))
            win_stage.append(stage_of[starts])
        self.win_file = np.concatenate(win_file)
        self.win_start = np.concatenate(win_start)
        self.win_kind = np.concatenate(win_kind)  # 0 walk, 1 crop, 2 clip
        self.win_stage = np.concatenate(win_stage)  # motion stage of the first frame

        keep = np.arange(0, len(self.win_start), window_stride)
        if max_windows is not None and len(keep) > max_windows:
            keep = keep[np.linspace(0, len(keep) - 1, max_windows).round().astype(np.int64)]
        for name in ("win_file", "win_start", "win_kind", "win_stage"):
            setattr(self, name, getattr(self, name)[keep])

        self._h5 = {}
        logger.info(
            f"HanoiClipDataset[{self.split}]: {len(self)} clips from {len(seg_keys)} segments "
            f"(fpc={frames_per_clip}, fstp={self.fstp} rows at {self.rate_hz} Hz)"
        )

    def __getstate__(self):
        # h5py handles cannot be pickled into spawned DataLoader workers; each worker reopens lazily
        state = self.__dict__.copy()
        state["_h5"] = {}
        return state

    def _file(self, f):
        if f not in self._h5:
            self._h5[f] = h5py.File(self.files[f], "r")
        return self._h5[f]

    def __len__(self):
        return len(self.win_start)

    def __getitem__(self, index):
        f = self._file(int(self.win_file[index]))
        rows = self.win_start[index] + np.arange(self.frames_per_clip, dtype=np.int64) * self.fstp
        states = hanoi_states(f["state"][rows], f["proprio"][rows])
        actions = poses_to_diffs(states)
        extrinsics = np.zeros((len(rows), 6), dtype=np.float64)  # one fixed camera: unused (use_extrinsics false)
        buffer = f["pixels"][rows]  # [T, 224, 224, 3] uint8; rows strictly increasing
        if self.transform is not None:
            buffer = self.transform(buffer)
        return buffer, actions, states, extrinsics, rows, index


def init_data(
    data_path,
    batch_size,
    frames_per_clip=8,
    fps=4,
    crop_size=256,
    rank=0,
    world_size=1,
    camera_views=None,
    stereo_view=False,
    drop_last=True,
    num_workers=8,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
):
    """Same signature and return as app/vjepa_droid/droid.py::init_data; `data_path` is a split archive (.npz)."""
    dataset = HanoiClipDataset(data_path, frames_per_clip=frames_per_clip, fps=fps, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    return loader, sampler


def init_eval_data(
    data_path,
    batch_size,
    frames_per_clip=8,
    fps=4,
    transform=None,
    window_stride=1,
    max_windows=None,
    rank=0,
    world_size=1,
    num_workers=4,
    pin_mem=True,
    collator=None,
    shuffle_seed=None,
):
    """Deterministic held-out clips, sharded rank::world_size without padding (no clip is counted twice).
    With shuffle_seed the visiting order is a fixed random permutation (the set of clips is unchanged)."""
    dataset = HanoiClipDataset(
        data_path,
        frames_per_clip=frames_per_clip,
        fps=fps,
        transform=transform,
        window_stride=window_stride,
        max_windows=max_windows,
    )
    shard = torch.utils.data.Subset(dataset, list(range(rank, len(dataset), world_size)))
    loader = torch.utils.data.DataLoader(
        shard,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=shuffle_seed is not None,
        generator=None if shuffle_seed is None else torch.Generator().manual_seed(shuffle_seed),
        drop_last=False,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )
    return loader, dataset

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__, force=True)


def l1(a, b):
    return torch.mean(torch.abs(a - b), dim=-1)


def round_small_elements(tensor, threshold):
    mask = torch.abs(tensor) < threshold
    new_tensor = tensor.clone()
    new_tensor[mask] = 0
    return new_tensor


def cem(
    context_frame,
    context_pose,
    goal_frame,
    world_model,
    rollout=1,
    cem_steps=100,
    momentum_mean=0.25,
    momentum_std=0.95,
    momentum_mean_gripper=0.15,
    momentum_std_gripper=0.15,
    samples=100,
    topk=10,
    verbose=False,
    maxnorm=0.05,
    axis={},
    objective=l1,
    close_gripper=None,
    gripper_std=1.0,
    gripper_clip=0.75,
    init_std=None,
    lo=None,
    hi=None,
    include_mean=False,
    exclusive_gripper=None,
    gripper_feasible=False,
    return_best=False,
    round_gripper=0.25,
):
    """
    :param context_frame: [B=1, T=1, HW, D]
    :param goal_frame: [B=1, T=1, HW, D]
    :param world_model: f(context_frame, action) -> next_frame [B, 1, HW, D]
    :return: [B=1, rollout, 7] an action trajectory over rollout horizon

    Cross-Entropy Method
    -----------------------
    1. for rollout horizon:
    1.1. sample several actions
    1.2. compute next states using WM
    3. compute similarity of final states to goal_frames
    4. select topk samples and update mean and std using topk action trajs
    5. choose final action to be mean of distribution

    Opt-in options (defaults: the behaviour above, unchanged), on the sampled [dx, dy, dz, gripper]:
    init_std          initial std per dim (4 values) instead of [maxnorm] * 3 + [gripper_std]
    lo, hi            per-dim clip box (4 values each) instead of +-maxnorm and +-gripper_clip
    include_mean      sample 0 is the current mean (JEPA-WMs planner)
    exclusive_gripper samples with |gripper| >= this move no xyz, the others change no gripper (the expert never
                      moves while opening or closing)
    gripper_feasible  clip the gripper change to what the rolled-forward closedness c allows: [-c, 1 - c]
    return_best       return the best-scoring sample of the last iteration instead of the elite mean
    round_gripper     final gripper values below this are zeroed (repo: 0.25; 0 turns it off)
    """
    context_frame = context_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    goal_frame = goal_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    context_pose = context_pose.repeat(samples, 1, 1)  # Reshape to [S, 1, 7]

    # Current estimate of the mean/std of distribution over action trajectories
    mean = torch.cat(
        [
            torch.zeros((rollout, 3), device=context_frame.device),
            torch.zeros((rollout, 1), device=context_frame.device),
        ],
        dim=-1,
    )

    std = torch.cat(
        [
            torch.ones((rollout, 3), device=context_frame.device) * maxnorm,
            torch.ones((rollout, 1), device=context_frame.device) * gripper_std,
        ],
        dim=-1,
    )
    if init_std is not None:
        std = torch.tensor(init_std, dtype=std.dtype, device=std.device).expand(rollout, 4).clone()
    if lo is not None:
        lo = torch.tensor(lo, dtype=std.dtype, device=std.device)
        hi = torch.tensor(hi, dtype=std.dtype, device=std.device)

    for ax in axis.keys():
        mean[:, ax] = axis[ax]

    def sample_action_traj():
        """Sample several action trajectories"""
        action_traj, frame_traj, pose_traj = None, context_frame, context_pose

        for h in range(rollout):

            # -- sample new action
            action_samples = torch.randn(samples, mean.size(1), device=mean.device) * std[h] + mean[h]
            if include_mean:
                action_samples[0] = mean[h]
            if lo is None:
                action_samples[:, :3] = torch.clip(action_samples[:, :3], min=-maxnorm, max=maxnorm)
                action_samples[:, -1:] = torch.clip(action_samples[:, -1:], min=-gripper_clip, max=gripper_clip)
            else:
                action_samples = torch.maximum(torch.minimum(action_samples, hi), lo)
            for ax in axis.keys():
                action_samples[:, ax] = axis[ax]
            if gripper_feasible:  # before the move-or-grip split, so "open while open" becomes a move sample
                c = pose_traj[:, -1, -1].to(action_samples.dtype)
                action_samples[:, -1] = torch.minimum(torch.maximum(action_samples[:, -1], -c), 1.0 - c)
            if exclusive_gripper is not None:
                grip = action_samples[:, -1].abs() >= exclusive_gripper
                action_samples[grip, :3] = 0.0
                action_samples[~grip, -1] = 0.0
            action_samples = torch.cat(
                [
                    action_samples[:, :3],
                    torch.zeros((len(action_samples), 3), device=mean.device),
                    action_samples[:, -1:],
                ],
                dim=-1,
            )[:, None]
            if close_gripper is not None and h >= close_gripper:
                action_samples[:, :, -1] = 1.0
            if gripper_feasible and close_gripper is not None and h >= close_gripper:  # held closed: close if open
                action_samples[:, 0, -1] = 1.0 - pose_traj[:, -1, -1].to(action_samples.dtype)

            action_traj = (
                torch.cat([action_traj, action_samples], dim=1) if action_traj is not None else action_samples
            )

            # -- compute next state
            next_frame, next_pose = world_model(frame_traj, action_traj, pose_traj)
            frame_traj = torch.cat([frame_traj, next_frame], dim=1)
            pose_traj = torch.cat([pose_traj, next_pose], dim=1)

        return action_traj, frame_traj

    best = {}

    def select_topk_action_traj(final_state, goal_state, actions):
        """Get the topk action trajectories that bring us closest to goal"""
        sims = objective(final_state.flatten(1), goal_state.flatten(1))
        indices = sims.topk(topk, largest=False).indices
        selected_actions = actions[indices]
        best["traj"] = actions[indices[0]]
        return selected_actions

    for step in tqdm(range(cem_steps), disable=True):
        action_traj, frame_traj = sample_action_traj()
        selected_actions = select_topk_action_traj(
            final_state=frame_traj[:, -1], goal_state=goal_frame, actions=action_traj
        )
        mean_selected_actions = selected_actions.mean(dim=0)
        std_selected_actions = selected_actions.std(dim=0)

        # -- Update new sampling mean and std based on the top-k samples
        mean = torch.cat(
            [
                mean_selected_actions[..., :3] * (1.0 - momentum_mean) + mean[..., :3] * momentum_mean,
                mean_selected_actions[..., -1:] * (1.0 - momentum_mean_gripper)
                + mean[..., -1:] * momentum_mean_gripper,
            ],
            dim=-1,
        )
        std = torch.cat(
            [
                std_selected_actions[..., :3] * (1.0 - momentum_std) + std[..., :3] * momentum_std,
                std_selected_actions[..., -1:] * (1.0 - momentum_std_gripper) + std[..., -1:] * momentum_std_gripper,
            ],
            dim=-1,
        )

        logger.info(f"new mean: {mean.sum(dim=0)} {std.sum(dim=0)}")

    out = best["traj"][:, [0, 1, 2, 6]] if return_best else mean
    new_action = torch.cat(
        [
            out[..., :3],
            torch.zeros((rollout, 3), device=mean.device),
            round_small_elements(out[..., -1:], round_gripper),
        ],
        dim=-1,
    )[None, :]

    return new_action


def cem_categorical(
    context_frame,
    context_pose,
    goal_frame,
    world_model,
    codebook,
    code_prior=None,
    rollout=1,
    samples=800,
    cem_steps=10,
    topk=10,
    alpha=0.7,
    prior_temp=0.5,
    max_enumerate=4096,
    objective=l1,
    axis={},
    close_gripper=None,
    gripper_feasible=True,
    low_floor=None,
    lateral_limit=0.003,
    box=None,
    **unused,
):
    """
    Planning over a finite action set (not part of the V-JEPA 2 code; added for the Hanoi planner tests).
    codebook: [K, 4] rows [dx, dy, dz, gripper]. Every planned step is one of these codes, so the world model is only
    queried with actions shaped like its training actions.
    - If K ** rollout <= max_enumerate, every code sequence is scored (exact search).
    - Otherwise categorical CEM: per step a distribution over codes, initialised from code_prior ** prior_temp
      (0: uniform; 1: the codes' training frequencies), updated toward the top-k elite frequencies
      (p <- (1 - alpha) p + alpha f), with the current mode kept as sample 0.
    axis (held dims), close_gripper (held closed from step h) and gripper_feasible (change within [-c, 1 - c] of the
    imagined closedness c) are applied to each code as in cem. low_floor (m, default off): a step more than
    lateral_limit sideways with an end below this height (the imagined z) scores +inf, i.e. the planner only ranks
    steps a low-lateral guard (robot_io) would execute. box ((lo, hi) xyz in m, default off): a step whose imagined
    end leaves the box scores +inf (the executor would clamp it). Returns the best-scoring sequence [1, rollout, 7].
    """
    dev = context_frame.device
    codes = torch.as_tensor(codebook, dtype=torch.float32, device=dev)
    K = codes.size(0)
    if box is not None:
        box_lo, box_hi = (torch.as_tensor(b, dtype=torch.float32, device=dev) for b in box)

    def score(idx):
        n = idx.size(0)
        frames, poses, acts = context_frame.repeat(n, 1, 1, 1), context_pose.repeat(n, 1, 1), None
        blocked = torch.zeros(n, dtype=torch.bool, device=dev)
        for h in range(rollout):
            a = codes[idx[:, h]].clone()
            for ax in axis.keys():
                a[:, ax] = axis[ax]
            if low_floor is not None:
                z = poses[:, -1, 2].to(a.dtype)
                blocked |= (torch.minimum(z, z + a[:, 2]) < low_floor) & (a[:, :2].norm(dim=-1) > lateral_limit)
            if box is not None:
                end = poses[:, -1, :3].to(a.dtype) + a[:, :3]
                blocked |= ((end < box_lo) | (end > box_hi)).any(dim=-1)
            c = poses[:, -1, -1].to(a.dtype)
            if gripper_feasible:
                a[:, 3] = torch.minimum(torch.maximum(a[:, 3], -c), 1.0 - c)
            if close_gripper is not None and h >= close_gripper:
                a[:, 3] = (1.0 - c) if gripper_feasible else 1.0
            a = torch.cat([a[:, :3], torch.zeros((n, 3), device=dev), a[:, 3:]], dim=-1)[:, None]
            acts = a if acts is None else torch.cat([acts, a], dim=1)
            next_frame, next_pose = world_model(frames, acts, poses)
            frames, poses = torch.cat([frames, next_frame], dim=1), torch.cat([poses, next_pose], dim=1)
        energy = objective(frames[:, -1].flatten(1), goal_frame.flatten(1)).float()
        return energy.masked_fill(blocked, float("inf")), acts

    def score_chunked(idx):
        out = [score(chunk) for chunk in idx.split(samples)]
        return torch.cat([e for e, _ in out]), torch.cat([a for _, a in out])

    if K**rollout <= max_enumerate:
        grids = torch.meshgrid(*[torch.arange(K, device=dev)] * rollout, indexing="ij")
        energy, acts = score_chunked(torch.stack([g.reshape(-1) for g in grids], dim=1))
        return acts[energy.argmin()][None]

    prior = torch.ones(K, device=dev) if code_prior is None else torch.as_tensor(code_prior, device=dev).float()
    p = (prior.clamp_min(1e-8) ** prior_temp).expand(rollout, K).clone()
    p /= p.sum(-1, keepdim=True)
    best_e, best_a = float("inf"), None
    for _ in range(cem_steps):
        idx = torch.stack([torch.multinomial(p[h], samples, replacement=True) for h in range(rollout)], dim=1)
        idx[0] = p.argmax(-1)
        energy, acts = score(idx)
        if float(energy.min()) < best_e:
            best_e, best_a = float(energy.min()), acts[energy.argmin()]
        elite = idx[energy.topk(topk, largest=False).indices]  # [topk, rollout]
        freq = torch.nn.functional.one_hot(elite, K).float().mean(0)  # [rollout, K]
        p = (1.0 - alpha) * p + alpha * freq
    return best_a[None]


def compute_new_pose(pose, action):
    """
    :param pose: [B, T=1, 7]
    :param action: [B, T=1, 7]
    :returns: [B, T=1, 7]
    """
    device, dtype = pose.device, pose.dtype
    pose = pose[:, 0].cpu().numpy()
    action = action[:, 0].cpu().numpy()
    # -- compute delta xyz
    new_xyz = pose[:, :3] + action[:, :3]
    # -- compute delta theta
    thetas = pose[:, 3:6]
    delta_thetas = action[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    delta_matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in delta_thetas]
    angle_diff = [delta_matrices[t] @ matrices[t] for t in range(len(matrices))]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    new_angle = np.stack([d for d in angle_diff], axis=0)  # [B, 7]
    # -- compute delta gripper
    new_closedness = pose[:, -1:] + action[:, -1:]
    new_closedness = np.clip(new_closedness, 0, 1)
    # -- new pose
    new_pose = np.concatenate([new_xyz, new_angle, new_closedness], axis=-1)
    return torch.from_numpy(new_pose).to(device).to(dtype)[:, None]


def poses_to_diff(start, end):
    """
    :param start: [7]
    :param end: [7]
    """
    try:
        start = start.numpy()
        end = end.numpy()
    except Exception:
        pass

    # --

    s_xyz = start[:3]
    e_xyz = end[:3]
    xyz_diff = e_xyz - s_xyz

    # --

    s_thetas = start[3:6]
    e_thetas = end[3:6]
    s_rotation = Rotation.from_euler("xyz", s_thetas, degrees=False).as_matrix()
    e_rotation = Rotation.from_euler("xyz", e_thetas, degrees=False).as_matrix()
    rotation_diff = e_rotation @ s_rotation.T
    theta_diff = Rotation.from_matrix(rotation_diff).as_euler("xyz", degrees=False)

    # --

    s_gripper = start[-1:]
    e_gripper = end[-1:]
    gripper_diff = e_gripper - s_gripper

    action = np.concatenate([xyz_diff, theta_diff, gripper_diff], axis=0)
    return torch.from_numpy(action)

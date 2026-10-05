# The V-JEPA 2-AC forward pass and loss of app/vjepa_droid/train.py (lines 408-441), factored out so training,
# validation and held-out evaluation compute exactly the same quantities. Only change: the batch size is read from
# the tensors instead of the config, so the last partial evaluation batch works.

import torch
import torch.nn.functional as F


def encode_clips(target_encoder, clips, normalize_reps):
    """clips [B, C, T, H, W] -> h [B, T * tokens_per_frame, D]; each frame is encoded alone as a 2-frame tubelet."""
    B, T = clips.size(0), clips.size(2)
    c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = target_encoder(c)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


def predict(predictor, z, actions, states, extrinsics, tokens_per_frame, auto_steps, normalize_reps):
    """Teacher-forced predictions z_tf (frames 1..T-1) and the auto-regressive rollout z_ar (frames 1..auto_steps)."""

    def _step_predictor(_z, _a, _s, _e):
        _z = predictor(_z, _a, _s, _e)
        if normalize_reps:
            _z = F.layer_norm(_z, (_z.size(-1),))
        return _z

    # -- one step of predictor with teacher forcing
    _z, _a, _s, _e = z[:, :-tokens_per_frame], actions, states[:, :-1], extrinsics[:, :-1]
    z_tf = _step_predictor(_z, _a, _s, _e)

    # -- full auto-regressive rollouts of predictor
    _z = torch.cat([z[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]], dim=1)
    for n in range(1, auto_steps):
        _a, _s, _e = actions[:, : n + 1], states[:, : n + 1], extrinsics[:, : n + 1]
        _z_nxt = _step_predictor(_z, _a, _s, _e)[:, -tokens_per_frame:]
        _z = torch.cat([_z, _z_nxt], dim=1)
    z_ar = _z[:, tokens_per_frame:]

    return z_tf, z_ar


def l1_loss(z, h, tokens_per_frame, loss_exp):
    _h = h[:, tokens_per_frame : z.size(1) + tokens_per_frame]
    return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp


def per_clip_loss(z, h, tokens_per_frame, loss_exp):
    """Same quantity as l1_loss, kept per clip ([B]) for evaluation breakdowns."""
    _h = h[:, tokens_per_frame : z.size(1) + tokens_per_frame]
    return (torch.abs(z - _h) ** loss_exp).mean(dim=(1, 2)) / loss_exp

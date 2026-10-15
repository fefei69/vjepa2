# Start a fine-tune from the JEPA-WMs corrected V-JEPA 2-AC (Terver et al., arXiv 2512.24497, Appendix C), not from the
# released vjepa2-ac-vitg.pt, whose 2-step rollout loss term was computed with the bug fixed in vjepa2 commit 93366fe.
#
#   predictor  droid_vj2ac_noprop.pth.tar ('predictor', keys 'module.*'): https://dl.fbaipublicfiles.com/jepa-wms/
#              (jepa-wms hubconf 'vjepa2_ac_droid'). Retrained on DROID from scratch with the fixed loss, without the
#              state token (proprio_tokens 0), so it needs model.use_state: false.
#   encoder    vitg.pt 'encoder' (the V-JEPA 2 ViT-g the JEPA-WMs run trained its predictor on, pretrain_enc_ckpt_key)
#
# The repo's load_pretrained loads with strict=False and only strips 'backbone.', so a prefix mismatch would load nothing
# and only log it. These loaders fail instead: every checkpoint tensor must land in the model, and the only model
# tensors allowed to stay at their initialisation are the ones listed in ALLOWED_MISSING.
import logging

import torch

logger = logging.getLogger()

# the extrinsics encoder exists in our predictor even with use_extrinsics false (unused in forward)
ALLOWED_MISSING = {"extrinsics_encoder.weight", "extrinsics_encoder.bias"}


def _unwrap(module):
    return module.module if hasattr(module, "module") else module


def _strip(state_dict, prefixes):
    out = {}
    for k, v in state_dict.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p) :]
        out[k] = v
    return out


def _load_strict(module, state_dict, allowed_missing, name):
    model_keys = set(_unwrap(module).state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    unexpected = sorted(ckpt_keys - model_keys)
    missing = sorted(model_keys - ckpt_keys)
    if unexpected or set(missing) - set(allowed_missing):
        raise RuntimeError(
            f"{name}: checkpoint does not match the model. unexpected ({len(unexpected)}): {unexpected[:8]}; "
            f"missing ({len(missing)}): {missing[:8]}"
        )
    msg = _unwrap(module).load_state_dict(state_dict, strict=False)  # shape mismatches still raise
    norm = sum(float(v.float().norm()) for v in state_dict.values() if torch.is_floating_point(v))
    logger.info(f"{name}: loaded {len(ckpt_keys)} tensors (sum of norms {norm:.1f}); left at init: {msg.missing_keys}")


def load_jepawms_init(encoder, target_encoder, predictor, encoder_path, encoder_key, predictor_path):
    """Context and target encoder <- encoder_path[encoder_key]; predictor <- predictor_path['predictor']. Strict."""
    ck = torch.load(encoder_path, map_location="cpu", mmap=True, weights_only=True)
    enc = _strip(ck[encoder_key], ("module.", "backbone."))
    _load_strict(encoder, enc, (), f"encoder <- {encoder_path}[{encoder_key}]")
    _load_strict(target_encoder, enc, (), f"target_encoder <- {encoder_path}[{encoder_key}]")
    del ck, enc
    ck = torch.load(predictor_path, map_location="cpu", mmap=True, weights_only=True)
    logger.info(f"predictor checkpoint {predictor_path}: epoch {ck.get('epoch')}")
    _load_strict(predictor, _strip(ck["predictor"], ("module.",)), ALLOWED_MISSING, f"predictor <- {predictor_path}")
    del ck
    return encoder, target_encoder, predictor


@torch.no_grad()
def compare_encoders(path_a, key_a, path_b, key_b):
    """Log whether two encoder state dicts are identical (e.g. vitg.pt 'encoder' vs vjepa2-ac-vitg.pt 'target_encoder')."""
    a = _strip(torch.load(path_a, map_location="cpu", mmap=True, weights_only=True)[key_a], ("module.", "backbone."))
    b = _strip(torch.load(path_b, map_location="cpu", mmap=True, weights_only=True)[key_b], ("module.", "backbone."))
    if set(a) != set(b):
        logger.info(f"encoder compare: key sets differ ({len(set(a) ^ set(b))} keys)")
        return False
    diff = {k: float((a[k].float() - b[k].float()).abs().max()) for k in a}
    worst = sorted(diff.items(), key=lambda kv: -kv[1])[:3]
    same = all(v == 0.0 for v in diff.values())
    logger.info(f"encoder compare {path_a}[{key_a}] vs {path_b}[{key_b}]: identical={same}; largest |diff| {worst}")
    return same

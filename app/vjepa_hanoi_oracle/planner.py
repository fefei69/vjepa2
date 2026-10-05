# ORACLE subgoal planning with a trained V-JEPA 2-AC checkpoint (privileged aids; see README.md -- not the baseline).
#
#   serve:    python -m app.vjepa_hanoi_oracle.planner serve --fname <yaml> --checkpoint <best.pt> \
#                 --subgoals <subgoals.npz>
#   validate: python -m app.vjepa_hanoi_oracle.planner validate --fname <yaml> --checkpoint <best.pt> \
#                 --subgoals <npz> \
#                 --archive <openpi play_test.npz> --max_rows 1000
#
# Each request plans ONE model step (0.267 s at 30 Hz / fps 4) toward a subgoal image of the current move
# (app/vjepa_hanoi_oracle/subgoals.py), steps capped at 20 mm per axis, with the repo planner
# (notebooks/utils/mpc_utils.py::cem) and x held at 0: x never moves in the training data, so the model cannot
# constrain it. The commanded target is the measured pose plus the planned [0, dy, dz], clamped to the workspace box;
# rotation is kept; gripper intent comes from the planned closedness. The goal latent is the mean latent of the
# subgoal's training exemplars.
#
# HTTP protocol (stdlib only): POST /plan with an .npz body holding
#   image (224, 224, 3) uint8  -- the camera frame after the recording's square_roi 360 -> 224 transform
#   pose  (6,) float           -- measured tool pose, xyz (m) + angle-axis (rad), as recorded in `state`
#   jaw   ()  float            -- measured gripper coordinate (as `proprio[:, 6]`)
#   subgoal str                -- "<before>><after>:<phase>", e.g. "AAAA>BAAA:descended"
# and receives JSON {target_pose, gripper_intent ("open"/"close"), action, goal_energy, plan_energy}. GET /health.

import argparse
import io
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import h5py
import numpy as np
import torch
import yaml

from app.vjepa_droid.transforms import make_transforms
from app.vjepa_hanoi.ac_step import encode_clips
from app.vjepa_hanoi.eval import STAGES, build_models, strip_prefixes
from app.vjepa_hanoi.hanoi import hanoi_states, poses_to_diffs
from app.vjepa_hanoi.plan_eval import load_mpc_utils, make_step_fn, rollout
from app.vjepa_hanoi_oracle.subgoals import STAGE_TO_PHASE, load, move_table

# Hanoi workspace from the recordings (x 0.414-0.500, y -0.058-0.090, z 0.057-0.192 m), padded by 5 mm
WORKSPACE_MIN = np.array([0.409, -0.063, 0.052])
WORKSPACE_MAX = np.array([0.505, 0.095, 0.197])


class HanoiPlanner:
    def __init__(
        self, fname, checkpoint, subgoals, max_exemplars=8, maxnorm=0.02, samples=400, cem_steps=10, device=None
    ):
        with open(fname) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        d = cfg["data"]
        self.crop = d.get("crop_size", 256)
        self.tpf = (self.crop // d["patch_size"]) ** 2
        self.normalize_reps = cfg["loss"]["normalize_reps"]
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.bfloat16 if str(cfg["meta"].get("dtype")).lower() == "bfloat16" else torch.float32
        self.encoder, self.predictor = build_models(cfg, self.device)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        self.encoder.load_state_dict(strip_prefixes(ckpt["target_encoder"]), strict=True)
        self.predictor.load_state_dict(strip_prefixes(ckpt["predictor"]), strict=True)
        del ckpt
        self.encoder.eval()
        self.predictor.eval()
        self.mpc = load_mpc_utils()
        self.step_fn = make_step_fn(self.predictor, self.tpf, self.normalize_reps, self.mpc.compute_new_pose)
        self.transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
            crop_size=self.crop,
        )
        self.cem_kw = dict(rollout=1, samples=samples, topk=10, cem_steps=cem_steps, momentum_mean=0.15,
                           momentum_std=0.15, maxnorm=maxnorm, axis={0: 0.0})  # fmt: skip
        self.library, paths = load(subgoals)
        self.files = [h5py.File(p, "r") for p in paths]
        self.max_exemplars = max_exemplars
        self._goal_cache = {}

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32)

    @torch.no_grad()
    def encode(self, images):
        """images [N, 224, 224, 3] uint8 -> latents [N, 1, HW, D] (each frame alone, as in training)."""
        clip = self.transform(np.asarray(images))[None].to(self.device)  # [1, C, N, H, W]
        with self._autocast():
            z = encode_clips(self.encoder, clip, self.normalize_reps).float()
        return z.view(len(images), 1, self.tpf, -1)

    def goal_latent(self, subgoal):
        if subgoal not in self._goal_cache:
            refs = self.library[subgoal]
            refs = [refs[i] for i in np.linspace(0, len(refs) - 1, min(len(refs), self.max_exemplars)).astype(int)]
            images = np.stack([self.files[f]["pixels"][r] for f, r in refs])
            self._goal_cache[subgoal] = self.encode(images).mean(dim=0, keepdim=True)
        return self._goal_cache[subgoal]

    @torch.no_grad()
    def plan(self, image, pose, jaw, subgoal, seed=0):
        state7 = hanoi_states(np.asarray(pose, dtype=np.float64)[None], np.full((1, 8), float(jaw)))
        z0 = self.encode(np.asarray(image)[None])
        zg = self.goal_latent(subgoal)
        s = torch.as_tensor(state7[None], dtype=torch.float32, device=self.device)  # [1, 1, 7]
        torch.manual_seed(seed)
        with self._autocast():
            action = self.mpc.cem(context_frame=z0, context_pose=s, goal_frame=zg, world_model=self.step_fn,
                                  **self.cem_kw)[0][0].float()  # fmt: skip
            final = rollout(self.step_fn, z0, s, action[None, None]).float()
        goal_energy = float(torch.mean(torch.abs(z0 - zg)))  # how far the current frame is from the subgoal
        plan_energy = float(torch.mean(torch.abs(final - zg)))
        a = action.cpu().numpy()
        target = np.asarray(pose, dtype=np.float64).copy()
        target[:3] = np.clip(target[:3] + a[:3], WORKSPACE_MIN, WORKSPACE_MAX)
        target[0] = float(pose[0])  # x held
        closedness = float(np.clip(state7[0, 6] + a[6], 0.0, 1.0))
        return {
            "target_pose": target.tolist(),
            "gripper_intent": "close" if closedness > 0.5 else "open",
            "action": a.tolist(),
            "goal_energy": goal_energy,
            "plan_energy": plan_energy,
        }


def serve(planner, host, port):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "subgoals": len(planner.library)})
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):
            if self.path != "/plan":
                return self._send(404, {"error": "unknown path"})
            try:
                z = np.load(io.BytesIO(self.rfile.read(int(self.headers["Content-Length"]))))
                t0 = time.time()
                out = planner.plan(z["image"], z["pose"], float(z["jaw"]), str(z["subgoal"]))
                out["seconds"] = time.time() - t0
                self._send(200, out)
            except Exception as e:  # report to the client instead of dropping the connection
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            pass

    print(f"planner serving on http://{host}:{port} ({len(planner.library)} subgoals)", flush=True)
    HTTPServer((host, port), Handler).serve_forever()  # single-threaded: one GPU plan at a time


def validate(planner, archive, max_rows, out):
    """Offline steering check on held-out walks: from a usable row inside a transfer, plan one step toward the
    move's next subgoal (training exemplars of the same transition) and compare with the expert's actual next step
    (state 8 rows later). Both rows must be usable observations."""
    moves, paths = move_table(archive)
    files = [h5py.File(p, "r") for p in paths]
    cand = []
    for mv in moves:
        usable = set(mv["rows"].tolist())
        for r, st in zip(mv["rows"], mv["stages"]):
            if int(st) in STAGE_TO_PHASE and int(r) + 8 in usable:
                cand.append((mv["file"], int(r), int(st), mv["key"]))
    pick = np.linspace(0, len(cand) - 1, min(max_rows, len(cand))).astype(int)
    rec = {k: [] for k in ("true", "plan", "stage", "goal_energy")}
    t0 = time.time()
    for n, i in enumerate(pick):
        f, r, st, key = cand[i]
        h = files[f]
        pose, jaw = h["state"][[r, r + 8]].astype(np.float64), h["proprio"][[r, r + 8], 6]
        true = poses_to_diffs(hanoi_states(pose, np.repeat(jaw[:, None], 8, axis=1)))[0]
        res = planner.plan(h["pixels"][r], pose[0], jaw[0], f"{key}:{STAGE_TO_PHASE[st]}", seed=n)
        rec["true"].append(true)
        rec["plan"].append(np.asarray(res["action"]))
        rec["stage"].append(st)
        rec["goal_energy"].append(res["goal_energy"])
        if (n + 1) % 100 == 0:
            print(f"{n + 1}/{len(pick)} rows, {(time.time() - t0) / (n + 1):.2f} s/row", flush=True)
    rec = {k: np.asarray(v) for k, v in rec.items()}
    np.savez_compressed(out.replace(".json", "_rows.npz"), **rec)

    def summary(m):
        t, p = rec["true"][m, 1:3] * 1e3, rec["plan"][m, 1:3] * 1e3  # y-z plane (x is held)
        mag, err = np.linalg.norm(t, axis=1), np.linalg.norm(p - t, axis=1)
        mov = mag >= 5
        cos = np.sum(t[mov] * p[mov], 1) / (mag[mov] * np.linalg.norm(p[mov], axis=1) + 1e-9)
        tg, pg = rec["true"][m, 6], rec["plan"][m, 6]
        ev = np.abs(tg) >= 0.25
        return {
            "rows": int(m.sum()),
            "moving_rows": int(mov.sum()),
            "moving_yz_err_mm_median": float(np.median(err[mov])) if mov.any() else float("nan"),
            "moving_zero_action_mm_median": float(np.median(mag[mov])) if mov.any() else float("nan"),
            "moving_direction_cos_median": float(np.median(cos)) if mov.any() else float("nan"),
            "moving_direction_cos_positive": float(np.mean(cos > 0)) if mov.any() else float("nan"),
            "static_spurious_mm_median": float(np.median(err[~mov])) if (~mov).any() else float("nan"),
            "gripper_event_rows": int(ev.sum()),
            "gripper_event_sign_accuracy": (
                float(np.mean(np.sign(pg[ev]) == np.sign(tg[ev]))) if ev.any() else float("nan")
            ),
        }

    result = {"archive": archive, "overall": summary(np.ones(len(rec["stage"]), bool))}
    result["by_stage"] = {STAGES[int(s)]: summary(rec["stage"] == s) for s in np.unique(rec["stage"])}
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    o = result["overall"]
    print(
        f"[subgoal steering] {o['rows']} rows ({o['moving_rows']} moving): y-z err "
        f"{o['moving_yz_err_mm_median']:.2f} mm vs zero-action {o['moving_zero_action_mm_median']:.2f}, cos "
        f"{o['moving_direction_cos_median']:.2f} (positive {o['moving_direction_cos_positive']:.2f}), static spurious "
        f"{o['static_spurious_mm_median']:.2f} mm, gripper events {o['gripper_event_sign_accuracy']:.2f} "
        f"({o['gripper_event_rows']}) -> {out}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("serve", "validate"))
    p.add_argument("--fname", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--subgoals", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--archive", default=None, help="validate: held-out split archive (play_test.npz)")
    p.add_argument("--max_rows", type=int, default=1000)
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--cem_steps", type=int, default=10)
    p.add_argument("--maxnorm", type=float, default=0.02, help="per-axis step cap (m); the expert moves ~13 mm/step")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    planner = HanoiPlanner(
        args.fname,
        args.checkpoint,
        args.subgoals,
        maxnorm=args.maxnorm,
        samples=args.samples,
        cem_steps=args.cem_steps,
    )
    if args.mode == "serve":
        serve(planner, args.host, args.port)
    else:
        validate(planner, args.archive, args.max_rows, args.out or f"{args.checkpoint}.subgoal_steering.json")


if __name__ == "__main__":
    main()

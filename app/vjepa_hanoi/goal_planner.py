# Plain V-JEPA 2-AC goal-image planning server (the baseline as published; no task-specific aids).
#
#   python -m app.vjepa_hanoi.goal_planner --fname <yaml> --checkpoint <best.pt> --host 0.0.0.0 --port 8766 \
#       [--goal_archive <openpi play_train.npz>]
#
# Each request plans with the repo planner (notebooks/utils/mpc_utils.py::cem) at the WorldModel defaults -- rollout 2,
# 400 samples, top-10, 10 iterations, maxnorm 0.05 m, rotation fixed at 0 -- from the current frame and measured state
# toward ONE goal image, and returns the first planned action. No subgoals, no task solver, no fixed axes, no step cap.
#
# The goal is either an image the client uploads (POST /goal: a photo of the goal configuration from the same camera
# after the square_roi transform), or a goal board (POST /goal with goal_board="CCCC"), for which the goal latent is
# the mean latent of training-split frames showing that board at the end of a move (arm parked, ring released).
#
# Execution safety, applied AFTER planning and reported in every reply (it never changes what the planner chose):
# the target is clamped to the recorded workspace box, and a step faster than --max_speed is scaled down.
#
# POST /plan with an .npz body {image (224,224,3) uint8, pose (6,) xyz + angle-axis, jaw ()} returns JSON
# {target_pose, gripper_intent, action, executed_delta, safety_scaled, goal_energy, plan_energy, seconds}.

import argparse
import io
import itertools
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import h5py
import numpy as np
import torch
import yaml

from app.vjepa_droid.transforms import make_transforms
from app.vjepa_hanoi.ac_step import encode_clips
from app.vjepa_hanoi.eval import build_models, strip_prefixes
from app.vjepa_hanoi.hanoi import hanoi_states
from app.vjepa_hanoi.plan_eval import load_mpc_utils, make_step_fn, rollout

WORKSPACE_MIN = np.array([0.409, -0.063, 0.052])  # recorded workspace (x 0.414-0.500, y -0.058-0.090,
WORKSPACE_MAX = np.array([0.505, 0.095, 0.197])  # z 0.057-0.192 m) padded by 5 mm
BOARDS = ["".join(b) for b in itertools.product("ABC", repeat=4)]
STEP_S = 8 / 30.0


def goal_board_rows(archive, board, max_rows=8):
    """Training rows showing `board` at the end of a move whose result is `board` (last usable row of the move)."""
    z = np.load(archive)
    obs, fi, seg, move, bidx = (z[k] for k in ("source_observation_indices", "file_indices", "segment_bounds",
                                               "move_indices", "board_indices"))  # fmt: skip
    target = BOARDS.index(board)
    last = {}
    for i in range(len(obs)):
        key = (int(fi[i]), int(seg[i, 0]), int(move[i]))
        if key not in last or obs[i] > obs[last[key]]:
            last[key] = i
    rows = [(int(fi[i]), int(obs[i])) for i in last.values() if bidx[i] == target]
    if not rows:
        raise ValueError(f"no training frame shows board {board}")
    pick = np.linspace(0, len(rows) - 1, min(max_rows, len(rows))).astype(int)
    return [rows[k] for k in pick], [str(p) for p in z["source_paths"]]


class GoalPlanner:
    def __init__(self, fname, checkpoint, goal_archive=None, max_speed=0.10, rollout_steps=2):
        with open(fname) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        d = cfg["data"]
        crop = d.get("crop_size", 256)
        self.tpf = (crop // d["patch_size"]) ** 2
        self.norm = cfg["loss"]["normalize_reps"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if str(cfg["meta"].get("dtype")).lower() == "bfloat16" else torch.float32
        self.encoder, self.predictor = build_models(cfg, self.device)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        self.encoder.load_state_dict(strip_prefixes(ckpt["target_encoder"]), strict=True)
        self.predictor.load_state_dict(strip_prefixes(ckpt["predictor"]), strict=True)
        del ckpt
        self.encoder.eval()
        self.predictor.eval()
        self.mpc = load_mpc_utils()
        self.step_fn = make_step_fn(self.predictor, self.tpf, self.norm, self.mpc.compute_new_pose)
        self.transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
            crop_size=crop,
        )
        self.cem_kw = dict(
            rollout=rollout_steps,
            samples=400,
            topk=10,
            cem_steps=10,
            momentum_mean=0.15,
            momentum_std=0.15,
            maxnorm=0.05,
        )  # fmt: skip  (world_model_wrapper.py defaults)
        self.goal_archive = goal_archive
        self.max_step = max_speed * STEP_S
        self.goal = None
        self.goal_desc = None

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32)

    @torch.no_grad()
    def encode(self, images):
        clip = self.transform(np.asarray(images))[None].to(self.device)
        with self._autocast():
            z = encode_clips(self.encoder, clip, self.norm).float()
        return z.view(len(images), 1, self.tpf, -1)

    def set_goal(self, image=None, board=None):
        if image is not None:
            self.goal, self.goal_desc = self.encode(np.asarray(image)[None]), "uploaded image"
        else:
            if self.goal_archive is None:
                raise ValueError("goal_board needs the server started with --goal_archive")
            refs, paths = goal_board_rows(self.goal_archive, board)
            images = []
            for f, r in refs:
                with h5py.File(paths[f], "r") as h:
                    images.append(h["pixels"][r])
            self.goal = self.encode(np.stack(images)).mean(dim=0, keepdim=True)
            self.goal_desc = f"board {board} ({len(refs)} training frames)"
        return self.goal_desc

    @torch.no_grad()
    def plan(self, image, pose, jaw, seed=0):
        if self.goal is None:
            raise ValueError("no goal set: POST /goal first")
        pose = np.asarray(pose, dtype=np.float64)
        s7 = hanoi_states(pose[None], np.full((1, 8), float(jaw)))
        z0 = self.encode(np.asarray(image)[None])
        s = torch.as_tensor(s7[None], dtype=torch.float32, device=self.device)
        torch.manual_seed(seed)
        with self._autocast():
            plan = self.mpc.cem(context_frame=z0, context_pose=s, goal_frame=self.goal, world_model=self.step_fn,
                                **self.cem_kw)[0].float()  # fmt: skip
            final = rollout(self.step_fn, z0, s, plan[None]).float()
        a = plan[0].cpu().numpy()  # execute the first planned action, then replan
        delta = a[:3].copy()
        norm = float(np.linalg.norm(delta))
        scaled = norm > self.max_step
        if scaled:
            delta *= self.max_step / norm
        target = pose.copy()
        target[:3] = np.clip(pose[:3] + delta, WORKSPACE_MIN, WORKSPACE_MAX)
        closedness = float(np.clip(s7[0, 6] + a[6], 0.0, 1.0))
        return {
            "target_pose": target.tolist(),
            "gripper_intent": "close" if closedness > 0.5 else "open",
            "action": a.tolist(),
            "executed_delta": (target[:3] - pose[:3]).tolist(),
            "safety_scaled": bool(scaled),
            "goal_energy": float(torch.mean(torch.abs(z0 - self.goal))),
            "plan_energy": float(torch.mean(torch.abs(final - self.goal))),
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
                self._send(200, {"ok": True, "goal": planner.goal_desc})
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):
            try:
                z = np.load(io.BytesIO(self.rfile.read(int(self.headers["Content-Length"]))))
                if self.path == "/goal":
                    board = str(z["goal_board"]) if "goal_board" in z.files else None
                    image = z["goal_image"] if "goal_image" in z.files else None
                    return self._send(200, {"goal": planner.set_goal(image=image, board=board)})
                if self.path == "/plan":
                    t0 = time.time()
                    out = planner.plan(z["image"], z["pose"], float(z["jaw"]))
                    out["seconds"] = time.time() - t0
                    return self._send(200, out)
                self._send(404, {"error": "unknown path"})
            except Exception as e:
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            pass

    print(f"plain goal planner serving on http://{host}:{port}", flush=True)
    HTTPServer((host, port), Handler).serve_forever()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fname", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--goal_archive", default=None, help="training split archive, enables goal_board goals")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--max_speed", type=float, default=0.10, help="execution safety limit (m/s), after planning")
    args = p.parse_args()
    serve(GoalPlanner(args.fname, args.checkpoint, args.goal_archive, args.max_speed), args.host, args.port)


if __name__ == "__main__":
    main()

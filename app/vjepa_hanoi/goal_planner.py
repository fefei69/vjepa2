# V-JEPA 2-AC goal-image planning server: the plain baseline, or the paper's pick-and-place protocol. No task aids.
#
#   python -m app.vjepa_hanoi.goal_planner --fname <yaml> --checkpoint <best.pt> --host 0.0.0.0 --port 8766 \
#       --goal_archive <training split .npz> [--protocol paper]
#
# Each request plans with the repo planner (notebooks/utils/mpc_utils.py::cem) from the current frame and measured
# state toward a goal image, and returns the first planned action (rotation fixed at 0):
#   --protocol plain (default)  the WorldModel defaults: rollout 2, 400 samples, top-10, 10 iterations, maxnorm 0.05 m;
#                               one goal image
#   --protocol paper            the V-JEPA 2 paper's robot settings (arXiv 2506.09985, sec. 4.2): planning horizon 1,
#                               800 samples, 10 refinement steps. Pick-and-place there uses two subgoal images ("object
#                               grasped", "object near the goal position") before the final goal, switched on a fixed
#                               time schedule; the client picks the active goal (arm_goal_client --protocol paper).
# There is no task solver, no fixed axis and no step cap.
#
# Goals (POST /goal, an .npz body):
#   goal_image (224,224,3)      one photo from the same camera after the square_roi transform
#   goal_images (n,224,224,3)   paper protocol: several photos in order (grasped, near the target, final)
#   goal_board "CCCC"           one training frame showing that board at the end of a move (arm retreated)
#   transition "AAAA>BAAA"      paper protocol, one game move: three frames of the first complete training move of that
#                               transition: the end of the grasp (ring grasped), the end of the transit (ring above the
#                               target peg) and the end of the release (ring placed; --final_stage 9: arm retreated)
# With --protocol paper the reply also gives the time schedule: the training split's median number of model steps the
# expert spends on each goal (app/vjepa_hanoi/paper_protocol_eval.py checks the protocol offline).
#
# Execution safety is applied AFTER planning and reported in every reply. It never changes what the planner chose.
# The target is clamped to the recorded workspace box, and a step faster than --max_speed is scaled down.
#
# POST /plan with an .npz body {image (224,224,3) uint8, pose (6,) xyz + angle-axis, jaw (), [goal_index ()]} returns
# JSON {target_pose, gripper_intent, action, executed_delta, safety_scaled, goal_index, goal_energy, plan_energy,
# seconds}.

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
from app.vjepa_hanoi.eval import build_models, strip_prefixes
from app.vjepa_hanoi.hanoi import hanoi_states
from app.vjepa_hanoi.paper_protocol_eval import goal_rows, moves, time_schedule
from app.vjepa_hanoi.plan_eval import load_mpc_utils, make_step_fn, rollout

WORKSPACE_MIN = np.array([0.409, -0.063, 0.052])  # recorded workspace (x 0.414-0.500, y -0.058-0.090,
WORKSPACE_MAX = np.array([0.505, 0.095, 0.197])  # z 0.057-0.192 m) padded by 5 mm
STEP_S = 8 / 30.0
PLANNERS = {  # notebooks/utils/world_model_wrapper.py defaults; the paper's robot settings
    "plain": dict(rollout=2, samples=400, topk=10, cem_steps=10, momentum_mean=0.15, momentum_std=0.15, maxnorm=0.05),
    "paper": dict(rollout=1, samples=800, topk=10, cem_steps=10, momentum_mean=0.15, momentum_std=0.15, maxnorm=0.05),
}


class GoalPlanner:
    def __init__(self, fname, checkpoint, goal_archive=None, max_speed=0.10, protocol="plain", final_stage=8):
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
        self.protocol = protocol
        self.cem_kw = PLANNERS[protocol]
        self.final_stage = final_stage
        self.goal_archive = goal_archive
        self.max_step = max_speed * STEP_S
        self.goals = []  # goal latents in order; /plan picks one by goal_index (default: the last)
        self.goal_desc = None
        self._train = None

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32)

    @torch.no_grad()
    def encode(self, images):
        clip = self.transform(np.asarray(images))[None].to(self.device)
        with self._autocast():
            z = encode_clips(self.encoder, clip, self.norm).float()
        return z.view(len(images), 1, self.tpf, -1)

    def training_moves(self):
        """Complete game moves of the training split (loaded once)."""
        if self.goal_archive is None:
            raise ValueError("training-frame goals need the server started with --goal_archive")
        if self._train is None:
            self._train = moves(self.goal_archive)[:2]
        return self._train

    def training_frame(self, file, row):
        with h5py.File(self.training_moves()[1][file], "r") as h:
            return h["pixels"][row]

    def schedule(self):
        """Paper protocol: median model steps the expert spends on goals 1, 2 and 3 (training split)."""
        reach = time_schedule(self.training_moves()[0], self.final_stage)
        return [reach[0], reach[1] - reach[0], reach[2] - reach[1]]

    def set_goal(self, images=None, board=None, transition=None):
        if (images is not None and len(images) > 1 or transition) and self.protocol != "paper":
            raise ValueError("subgoal lists are the paper protocol: start the server with --protocol paper")
        if images is not None:
            self.goals = [self.encode(np.asarray(im)[None]) for im in images]
            desc = f"{len(images)} uploaded image(s)"
        elif board is not None:
            train, _ = self.training_moves()
            m = next((m for m in train if m["transition"].endswith(">" + board)), None)
            if m is None:
                raise ValueError(f"no complete training move ends in board {board}")
            row = int(m["rows"][-1])
            self.goals = [self.encode(self.training_frame(m["file"], row)[None])]
            desc = f"board {board}: end of training move {m['transition']} (file {m['file']}, row {row})"
        elif transition is not None:
            train, _ = self.training_moves()
            m = next((m for m in train if m["transition"] == transition), None)
            if m is None:
                raise ValueError(f"no complete training move {transition} (one legal ring transfer, e.g. AAAA>BAAA)")
            g = goal_rows(m, self.final_stage)
            self.goals = [self.encode(self.training_frame(m["file"], g[k])[None]) for k in (1, 2, 3)]
            desc = (f"transition {transition}: training frames (file {m['file']}) ring grasped (row {g[1]}), above "
                    f"the target (row {g[2]}), final = end of stage {self.final_stage} (row {g[3]})")  # fmt: skip
        else:
            raise ValueError("give goal_image(s), goal_board or transition")
        self.goal_desc = desc
        out = {"goal": desc, "n_goals": len(self.goals), "protocol": self.protocol, "planner": self.cem_kw}
        if self.protocol == "paper" and self.goal_archive is not None:
            out["schedule"] = self.schedule()
        return out

    @torch.no_grad()
    def plan(self, image, pose, jaw, goal_index=-1, seed=0):
        if not self.goals:
            raise ValueError("no goal set: POST /goal first")
        goal = self.goals[goal_index]
        pose = np.asarray(pose, dtype=np.float64)
        s7 = hanoi_states(pose[None], np.full((1, 8), float(jaw)))
        z0 = self.encode(np.asarray(image)[None])
        s = torch.as_tensor(s7[None], dtype=torch.float32, device=self.device)
        torch.manual_seed(seed)
        with self._autocast():
            plan = self.mpc.cem(context_frame=z0, context_pose=s, goal_frame=goal, world_model=self.step_fn,
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
            "goal_index": goal_index % len(self.goals),
            "goal_energy": float(torch.mean(torch.abs(z0 - goal))),
            "plan_energy": float(torch.mean(torch.abs(final - goal))),
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
                self._send(200, {"ok": True, "protocol": planner.protocol, "planner": planner.cem_kw,
                                 "goal": planner.goal_desc, "n_goals": len(planner.goals)})  # fmt: skip
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):
            try:
                z = np.load(io.BytesIO(self.rfile.read(int(self.headers["Content-Length"]))))
                if self.path == "/goal":
                    images = list(z["goal_images"]) if "goal_images" in z.files else None
                    if images is None and "goal_image" in z.files:
                        images = [z["goal_image"]]
                    board = str(z["goal_board"]) if "goal_board" in z.files else None
                    transition = str(z["transition"]) if "transition" in z.files else None
                    return self._send(200, planner.set_goal(images=images, board=board, transition=transition))
                if self.path == "/plan":
                    t0 = time.time()
                    gi = int(z["goal_index"]) if "goal_index" in z.files else -1
                    out = planner.plan(z["image"], z["pose"], float(z["jaw"]), goal_index=gi)
                    out["seconds"] = time.time() - t0
                    return self._send(200, out)
                self._send(404, {"error": "unknown path"})
            except Exception as e:
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            pass

    print(f"{planner.protocol} goal planner {planner.cem_kw} serving on http://{host}:{port}", flush=True)
    HTTPServer((host, port), Handler).serve_forever()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fname", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--goal_archive", default=None, help="training split archive: goal_board / transition goals")
    p.add_argument("--protocol", choices=tuple(PLANNERS), default="plain")
    p.add_argument("--final_stage", type=int, choices=(8, 9), default=8, help="paper protocol: final goal frame")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--max_speed", type=float, default=0.10, help="execution safety limit (m/s), after planning")
    args = p.parse_args()
    planner = GoalPlanner(args.fname, args.checkpoint, args.goal_archive, args.max_speed, args.protocol,
                          args.final_stage)  # fmt: skip
    serve(planner, args.host, args.port)


if __name__ == "__main__":
    main()

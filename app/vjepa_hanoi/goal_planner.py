# V-JEPA 2-AC goal-image planning server: the plain baseline, or the paper's pick-and-place protocol. No task aids.
#
#   python -m app.vjepa_hanoi.goal_planner --fname <yaml> --checkpoint <best.pt> --host 0.0.0.0 --port 8766 \
#       --goal_archive <training split .npz> [--protocol paper]
#   on the robot machine, from a bundle written by app/vjepa_hanoi/export_robot_bundle.py (no recordings needed):
#   python -m app.vjepa_hanoi.goal_planner --fname <bundle>/config.yaml --checkpoint <bundle>/weights.pt \
#       --goal_bundle <bundle>/goals.npz --protocol paper --host 127.0.0.1 --port 8766
#
# Each request plans with the repo planner (notebooks/utils/mpc_utils.py::cem) from the current frame and measured
# state toward a goal image, and returns the first planned action (rotation fixed at 0):
#   --protocol plain (default)  the WorldModel defaults: rollout 2, 400 samples, top-10, 10 iterations, maxnorm 0.05 m;
#                               one goal image
#   --protocol paper            the V-JEPA 2 paper's robot settings (arXiv 2506.09985, sec. 4.2): planning horizon 1,
#                               800 samples, 10 refinement steps. Pick-and-place there uses two subgoal images ("object
#                               grasped", "object near the goal position") before the final goal, switched on a fixed
#                               time schedule; the client picks the active goal (arm_goal_client --protocol paper).
# There is no task solver and no step cap. x is planned freely unless --fix_x (off by default; a disclosed deviation).
# x stays on the peg line throughout the play data, so a play-trained model cannot learn it and plans it at random;
# fix_x holds it there. Do NOT use fix_x from the episode start pose: in the one-move data (first moves) the arm
# starts 8 cm back and moves x from 0.414 to 0.498 m during the approach, so holding x would stop it short of the pegs.
# --hold_grasp (paper protocol, off by default; a disclosed deviation): while goal 2 (ring above the target) is active,
# the gripper is held closed with the repo planner's own close_gripper option. Without it, the planner commands
# "open" on most carrying steps (paper_protocol/SUMMARY.md) and drops the ring.
# The two server flags are defaults. A /goal request may override them for that run (fix_x, hold_grasp), so one
# server can serve every variant of a test session. Every reply reports the options in force.
# --predictor_chunk N (off by default) passes CEM's samples through the predictor N at a time. The per-sample math is
# unchanged (chunks of 100 and 200 gave bit-identical plans); only the peak GPU memory drops. One batch of 800 runs
# out of memory on a 16 GB GPU; 200 with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True peaks at 12.4 GiB (RTX 5080,
# about 25 s per step).
# --rollout H / --samples N (off by default; a disclosed deviation from the protocol's horizon 1 and 800 samples)
# override the planner. Horizon 1 is greedy: from the hover it closes the jaws at once, because that changes the next
# frame most toward "ring grasped". On a 16 GB GPU, H >= 2 needs --samples 400 and --predictor_chunk 100 (H=3: about
# 80 s per step on an RTX 5080, under the client's 120 s timeout).
# Options for the descend/grasp failure seen on the arm (robot_results/2026-10-13/RESULTS.md), off by default; the
# server flags are defaults and a /goal request may override hold_open, hold_xy and subgoals per run:
#   planner deviations: --gripper_std / --gripper_clip (CEM's initial gripper std and clip; repo 1.0 and 0.75);
#                       --subgoals stages (one goal per motion stage: the ends of descend, grasp, lift, transit, insert
#                       and the final stage, from the same training demo, on the demos' median timing; needs a goal
#                       bundle exported with stage frames, or --goal_archive)
#   task aids:          --hold_open (gripper action held at 0 while the descend goal is active; --subgoals stages only)
#                       --hold_xy (x/y actions held at 0 while a descend or grasp goal is active; with the paper's three
#                       goals that is goal 1, which from the episode start also covers the approach: hover start only)
#                       with --subgoals stages, --hold_grasp holds the gripper closed during lift, transit and insert
# The rule per goal is paper_protocol_eval.aids, the same as offline.
# Planner presets, off by default (planner deviations; the same settings as paper_protocol_eval --protocol/--sampling):
#   --planner jepawms   the JEPA-WMs V-JEPA 2-AC planner: 300 x 15, top 10, horizon 3, no momentum, std 0.1, clip
#                       +-0.1 / +-0.75, mean kept as a sample, L2 energy, predictor context = the last 2 frames
#                       (rotation stays 0; the arm still executes one action and replans)
#   --sampling data     samples kept inside the training actions' range at the model step: from --goal_archive, or
#                       from the bundle (export_robot_bundle stores the bounds in goals.npz)
#   --sampling cat      categorical planning over a codebook of training actions (stay, close, open, 48 k-means moves;
#                       paper_protocol_eval header), from --goal_archive or the bundle; a planned step longer than
#                       the cap (--max_speed x the model step) is scaled down when executed, as for every planner
#                       (--limit_codes_to_cap instead drops those codes; offline it fails at fps 1 with 36 mm);
#                       --guard_floor_mm 166.13 (robot_io's low floor) also drops, per step, the codes more than 3 mm
#                       sideways with an end below that height, which the low-lateral guard would abort;
#                       --box_feasible drops the codes whose end leaves the workspace box (the clamp would undo them)
#
# Goals (POST /goal, an .npz body; optional fix_x () and hold_grasp () booleans override the server defaults):
#   goal_image (224,224,3)      one photo from the same camera after the square_roi transform
#   goal_images (n,224,224,3)   paper protocol: several photos in order (grasped, near the target, final)
#   goal_board "CCCC"           one training frame showing that board at the end of a move (arm retreated)
#   transition "AAAA>BAAA"      paper protocol, one game move: three frames of the first complete training move of that
#                               transition: the end of the grasp (ring grasped), the end of the transit (ring above the
#                               target peg) and the end of the release (ring placed; --final_stage 9: arm retreated)
# With --protocol paper the reply also gives the time schedule: the median number of model steps the expert spends on
# each goal, over the training demos of the same transition (over all training moves for uploaded images or a
# board). app/vjepa_hanoi/paper_protocol_eval.py checks the protocol offline with the same rule.
#
# Execution safety is applied AFTER planning and reported in every reply. It never changes what the planner chose.
# The target is clamped to the workspace box of the goal archive's training data (x, y and the top padded by 5 mm; never
# more than 1 mm below the lowest recorded tool height), then a step faster than --max_speed is scaled down, so no
# executed step is longer than max_speed x the model step (0.267 s at fps 4). The orientation is always the recorded
# one (RECORDED_ROTVEC). A measured pose more than 10 mm outside the box or 3 deg off that orientation is refused (HTTP
# 400), not moved from.
#
# The model step follows the config's data.fps (8 raw rows at fps 4, 30 at fps 1): the schedules, the step cap and, for
# a model trained at another fps, the planner's maxnorm box (0.05 m per fps-4 step, scaled with the step length unless
# --maxnorm is given) all use it, and /health and /goal report it (step_rows, step_s).
#
# POST /plan with an .npz body {image (224,224,3) uint8, pose (6,) xyz + angle-axis, jaw (), [goal_index ()]} returns
# JSON {target_pose, gripper_intent, action, executed_delta, safety_scaled, goal_index, goal_energy, plan_energy,
# seconds}.

import argparse
import io
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import h5py
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from app.vjepa_droid.transforms import make_transforms
from app.vjepa_hanoi.ac_step import encode_clips
from app.vjepa_hanoi.eval import build_models, strip_prefixes
from app.vjepa_hanoi.hanoi import RATE_HZ, hanoi_states, step_rows
from app.vjepa_hanoi.paper_protocol_eval import (
    FSTP,
    aids,
    JEPAWMS,
    JEPAWMS_WINDOW,
    data_sampling,
    default_maxnorm,
    goal_rows,
    goal_stages,
    moves,
    l2,
    time_schedule,
)
from app.vjepa_hanoi.plan_eval import load_mpc_utils, make_step_fn, rollout, windowed

# fallback box when the goal archive has no measured_xyz: the six-task recordings (x 0.414-0.500, y -0.058-0.090,
# z 0.057-0.192 m) padded by 5 mm, except below (the bottom ring's grasp height, 3 mm above the table)
WORKSPACE_MIN = np.array([0.409, -0.063, 0.057])
WORKSPACE_MAX = np.array([0.505, 0.095, 0.197])
RECORDED_ROTVEC = np.array([0.0, 0.7854, 0.0])  # the only orientation the recordings command (= START_POSE[3:6])
OUTSIDE_TOL_M, ROT_TOL_RAD = 0.010, np.deg2rad(3.0)
PLANNERS = {  # notebooks/utils/world_model_wrapper.py defaults; the paper's robot settings
    "plain": dict(rollout=2, samples=400, topk=10, cem_steps=10, momentum_mean=0.15, momentum_std=0.15, maxnorm=0.05),
    "paper": dict(rollout=1, samples=800, topk=10, cem_steps=10, momentum_mean=0.15, momentum_std=0.15, maxnorm=0.05),
}


def chunked(step_fn, n):
    """step_fn over the sample batch in slices of n: the same per-sample math with a lower peak GPU memory."""

    def step(reps, actions, poses):
        if reps.size(0) <= n:
            return step_fn(reps, actions, poses)
        outs = [step_fn(r, a, p) for r, a, p in zip(reps.split(n), actions.split(n), poses.split(n))]
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])

    return step


class GoalPlanner:
    def __init__(self, fname, checkpoint, goal_archive=None, max_speed=0.10, protocol="plain", final_stage=8,
                 fix_x=False, hold_grasp=False, goal_bundle=None, predictor_chunk=0, planner_overrides=None,
                 hold_open=False, hold_xy=False, subgoals="paper3", planner="protocol", sampling="box",
                 guard_floor_mm=None, limit_codes_to_cap=False, box_feasible=False):
        if guard_floor_mm is not None and sampling != "cat":
            raise ValueError("--guard_floor_mm is a --sampling cat option")
        self.guard_floor = None if guard_floor_mm is None else guard_floor_mm / 1e3
        if box_feasible and sampling != "cat":
            raise ValueError("--box_feasible is a --sampling cat option")
        self.box_feasible = box_feasible
        self.planner_overrides = {k: v for k, v in (planner_overrides or {}).items() if v is not None}
        if planner == "jepawms" and self.planner_overrides:
            raise ValueError("--planner jepawms fixes horizon, samples, box and gripper std/clip")
        self.planner_preset, self.sampling = planner, sampling
        with open(fname) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        d = cfg["data"]
        self.fstp = step_rows(d.get("fps", 4))  # raw rows per model step
        self.step_s = self.fstp / RATE_HZ
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
        for m in (self.encoder, self.predictor):  # frozen: autocast then caches no bf16 copy of the weights (~0.6 GB)
            m.requires_grad_(False)
        self.mpc = load_mpc_utils()
        self.step_fn = make_step_fn(self.predictor, self.tpf, self.norm, self.mpc.compute_new_pose)
        self.objective = self.mpc.l1
        if planner == "jepawms":
            self.step_fn, self.objective = windowed(self.step_fn, JEPAWMS_WINDOW), l2
        self.predictor_chunk = predictor_chunk
        if predictor_chunk:
            self.step_fn = chunked(self.step_fn, predictor_chunk)
        self.transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
            crop_size=crop,
        )
        self.checkpoint = checkpoint
        self.protocol = protocol
        self.defaults = {"fix_x": bool(fix_x), "hold_grasp": bool(hold_grasp), "hold_open": bool(hold_open),
                         "hold_xy": bool(hold_xy), "subgoals": subgoals}  # fmt: skip
        self.final_stage = final_stage
        self.goal_archive = goal_archive
        self.max_step = max_speed * self.step_s
        self.box = (WORKSPACE_MIN, WORKSPACE_MAX, "six-task recordings (fallback)")
        if goal_archive is not None:
            z = np.load(goal_archive)
            if "measured_xyz" in z.files:
                xyz = z["measured_xyz"]
                lo, hi = xyz.min(0) - np.array([0.005, 0.005, 0.001]), xyz.max(0) + 0.005
                self.box = (lo, hi, f"training data of {goal_archive}")
        self.bundle = None
        if goal_bundle is not None:  # exported goal frames, schedules and box: no recordings needed
            if goal_archive is not None:
                raise ValueError("give --goal_archive or --goal_bundle, not both")
            b = np.load(goal_bundle)
            self.bundle = {str(t): i for i, t in enumerate(b["transitions"])}
            self.bundle_data = {k: b[k] for k in b.files}
            if int(b["final_stage"]) != final_stage:
                raise ValueError(f"the bundle's final goal is stage {int(b['final_stage'])}, not {final_stage}")
            bundle_rows = int(b["step_rows"]) if "step_rows" in b.files else FSTP  # bundles before fps support: 8
            if bundle_rows != self.fstp:
                raise ValueError(f"the bundle's schedules count {bundle_rows}-row steps, the model's step is "
                                 f"{self.fstp} rows (data.fps of {fname})")  # fmt: skip
            self.box = (b["box_min"], b["box_max"], f"training data of {str(b['goal_archive'])} (bundle)")
        self.goal_bundle = goal_bundle
        self.action_bounds = self.codebook = None
        if sampling == "cat":
            if self.bundle is not None:
                if "action_codebook_json" not in self.bundle_data:
                    raise ValueError("this bundle has no action codebook: re-export it (export_robot_bundle --goals_only)")
                self.codebook = json.loads(str(self.bundle_data["action_codebook_json"]))
            elif goal_archive is not None:
                from app.vjepa_hanoi.action_bounds import action_codebook

                self.codebook = action_codebook(goal_archive, d.get("fps", 4))
            else:
                raise ValueError("--sampling cat needs --goal_archive or --goal_bundle")
            if int(self.codebook["step_rows"]) != self.fstp:
                raise ValueError(f"the codebook is for {self.codebook['step_rows']}-row steps, the model's step is "
                                 f"{self.fstp} rows")  # fmt: skip
            if limit_codes_to_cap:  # opt-in: plan only steps the safety cap below executes unscaled. Offline
                # (descent_replay) this fails at fps 1 with a 36 mm cap: the planner moves sideways at the hover
                from app.vjepa_hanoi.action_bounds import limit_codebook

                n_codes = len(self.codebook["codes"])
                self.codebook = limit_codebook(self.codebook, self.max_step)
                print(f"codebook: {len(self.codebook['codes'])} of {n_codes} codes within the "
                      f"{self.max_step * 1e3:.1f} mm step cap")  # fmt: skip
        if sampling == "data":
            if self.bundle is not None:
                if "action_bounds_json" not in self.bundle_data:
                    raise ValueError("this bundle has no action bounds: re-export it (export_robot_bundle --goals_only)")
                self.action_bounds = json.loads(str(self.bundle_data["action_bounds_json"]))
            elif goal_archive is not None:
                from app.vjepa_hanoi.action_bounds import action_bounds

                self.action_bounds = action_bounds(goal_archive, d.get("fps", 4))
            else:
                raise ValueError("--sampling data needs --goal_archive or --goal_bundle")
            if int(self.action_bounds["step_rows"]) != self.fstp:
                raise ValueError(f"action bounds are for {self.action_bounds['step_rows']}-row steps, the model's "
                                 f"step is {self.fstp} rows")  # fmt: skip
        self.set_options(**self.defaults)
        self.schedule_now = None
        self.goals = []  # goal latents in order; /plan picks one by goal_index (default: the last)
        self.goal_desc = None
        self._train = None

    def set_options(self, fix_x, hold_grasp, hold_open=False, hold_xy=False, subgoals="paper3"):
        """The run's deviations (all off by default): hold x at 0, hold the gripper closed while carrying, hold it
        open during the descend goal, hold x/y during the descend and grasp goals, one goal per motion stage."""
        if (hold_grasp or hold_open or hold_xy or subgoals != "paper3") and self.protocol != "paper":
            raise ValueError("hold_grasp / hold_open / hold_xy / subgoals are paper-protocol options: start the server "
                             "with --protocol paper")  # fmt: skip
        if subgoals not in ("paper3", "stages"):
            raise ValueError(f"subgoals must be paper3 or stages, not {subgoals}")
        if hold_open and subgoals != "stages":
            raise ValueError("hold_open acts on the descend goal: use subgoals stages")
        self.options = {"fix_x": bool(fix_x), "hold_grasp": bool(hold_grasp), "hold_open": bool(hold_open),
                        "hold_xy": bool(hold_xy), "subgoals": subgoals}  # fmt: skip
        base = dict(PLANNERS[self.protocol])
        if self.fstp != FSTP:  # a model trained at another fps: the same speed box per step (default_maxnorm)
            base["maxnorm"] = default_maxnorm(self.fstp)
        if self.planner_preset == "jepawms":  # its fixed 0.1 box, not scaled with the step
            base = dict(JEPAWMS)
        self.cem_kw = base | self.planner_overrides  # --rollout / --samples / --maxnorm etc., if given
        if self.sampling == "data":
            self.cem_kw |= data_sampling(self.action_bounds)
        if self.sampling == "cat":
            self.cem_kw = {k: self.cem_kw[k] for k in ("rollout", "samples", "cem_steps", "topk")} | {
                "codebook": self.codebook["codes"], "code_prior": self.codebook["prior"], "prior_temp": 0.5}
            if self.guard_floor is not None:  # rank only steps robot_io's low-lateral guard executes
                self.cem_kw["low_floor"] = self.guard_floor
            if self.box_feasible:  # rank only steps that end inside the workspace box (the clamp below)
                self.cem_kw["box"] = (np.asarray(self.box[0]).tolist(), np.asarray(self.box[1]).tolist())
            if fix_x:
                self.cem_kw["axis"] = {0: 0.0}
        if fix_x:  # disclosed deviation: x stays on the peg line in the play data, so a play model cannot learn it
            self.cem_kw["axis"] = {0: 0.0}  # the repo planner's own option for holding an action dimension

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

    def schedule(self, transition=None, stages=None, start="move"):
        """Paper protocol: median model steps the expert spends on each goal, over the training demos of
        `transition` (all training moves if None or if it has none), from the move's first row or from the hover."""
        train = self.training_moves()[0]
        same = [m for m in train if m["transition"] == transition]
        reach = time_schedule(same or train, self.final_stage, stages, self.fstp, start)
        return np.diff([0] + reach).tolist()

    def set_goal(self, images=None, board=None, transition=None, fix_x=None, hold_grasp=None, hold_open=None,
                 hold_xy=None, subgoals=None):
        if (images is not None and len(images) > 1 or transition) and self.protocol != "paper":
            raise ValueError("subgoal lists are the paper protocol: start the server with --protocol paper")
        given = {"fix_x": fix_x, "hold_grasp": hold_grasp, "hold_open": hold_open, "hold_xy": hold_xy,
                 "subgoals": subgoals}  # fmt: skip
        self.set_options(**{k: self.defaults[k] if v is None else v for k, v in given.items()})
        stages_mode = self.options["subgoals"] == "stages"
        if stages_mode and transition is None:
            raise ValueError("subgoals stages needs a transition (its training demo's stage ends)")
        gst = goal_stages(self.options["subgoals"], self.final_stage)
        self.goal_stages = None  # motion stage of each goal frame, for the per-goal options (aids)
        if images is not None:
            self.goals = [self.encode(np.asarray(im)[None]) for im in images]
            desc = f"{len(images)} uploaded image(s)"
            if len(images) == 3 and self.protocol == "paper":  # the paper's three: grasped, near the target, final
                self.goal_stages = list(gst)
        elif board is not None:
            train, _ = self.training_moves()
            m = next((m for m in train if m["transition"].endswith(">" + board)), None)
            if m is None:
                raise ValueError(f"no complete training move ends in board {board}")
            row = int(m["rows"][-1])
            self.goals = [self.encode(self.training_frame(m["file"], row)[None])]
            desc = f"board {board}: end of training move {m['transition']} (file {m['file']}, row {row})"
        elif transition is not None and self.bundle is not None and stages_mode:
            if transition not in self.bundle:
                raise ValueError(f"the goal bundle has no move {transition}; it has {sorted(self.bundle)}")
            if "stage_frames" not in self.bundle_data:
                raise ValueError("this goal bundle has no stage frames: re-export it (export_robot_bundle --goals_only)")
            i = self.bundle[transition]
            if [int(x) for x in self.bundle_data["stage_goal_stages"]] != list(gst):
                raise ValueError(f"bundle stage goals {self.bundle_data['stage_goal_stages']} != {gst}")
            self.goals = [self.encode(f[None]) for f in self.bundle_data["stage_frames"][i]]
            self.goal_stages = list(gst)
            desc = (f"transition {transition}: bundled training frames (file "
                    f"{self.bundle_data['stage_frame_sources'][i][0][0]}), ends of stages {list(gst)} (rows "
                    f"{[int(r) for _, r in self.bundle_data['stage_frame_sources'][i]]})")  # fmt: skip
        elif transition is not None and self.bundle is not None:
            if transition not in self.bundle:
                raise ValueError(f"the goal bundle has no move {transition}; it has {sorted(self.bundle)}")
            i = self.bundle[transition]
            self.goal_stages = list(gst)
            self.goals = [self.encode(self.bundle_data["frames"][i, k][None]) for k in range(3)]
            src = self.bundle_data["frame_sources"][i]
            desc = (f"transition {transition}: bundled training frames (file {src[0][0]}) ring grasped (row "
                    f"{src[0][1]}), above the target (row {src[1][1]}), final = end of stage {self.final_stage} "
                    f"(row {src[2][1]})")  # fmt: skip
        elif transition is not None:
            train, _ = self.training_moves()
            m = next((m for m in train if m["transition"] == transition), None)
            if m is None:
                raise ValueError(f"no complete training move {transition} (one legal ring transfer, e.g. AAAA>BAAA)")
            g = goal_rows(m, self.final_stage, gst)
            self.goals = [self.encode(self.training_frame(m["file"], g[k])[None]) for k in range(1, len(gst) + 1)]
            self.goal_stages = list(gst)
            if stages_mode:
                desc = (f"transition {transition}: training frames (file {m['file']}), ends of stages {list(gst)} "
                        f"(rows {list(g.values())})")  # fmt: skip
            else:
                desc = (f"transition {transition}: training frames (file {m['file']}) ring grasped (row {g[1]}), "
                        f"above the target (row {g[2]}), final = end of stage {self.final_stage} (row {g[3]})")
        else:
            raise ValueError("give goal_image(s), goal_board or transition")
        self.goal_desc = desc
        out = {"goal": desc, "n_goals": len(self.goals), "protocol": self.protocol, "planner": self.cem_kw,
               "options": self.options}  # fmt: skip
        if self.protocol == "paper" and self.bundle is not None:
            i = self.bundle.get(transition)
            if stages_mode:
                out["schedule"] = self.bundle_data["stage_schedules"][i].tolist()
            else:
                out["schedule"] = (self.bundle_data["schedules"][i] if i is not None
                                   else self.bundle_data["global_schedule"]).tolist()  # fmt: skip
            key = "stage_hover_schedules" if stages_mode else "hover_schedules"
            if i is not None and key in self.bundle_data:  # the same medians counted from each demo's hover
                out["hover_schedule"] = self.bundle_data[key][i].tolist()
        elif self.protocol == "paper" and self.goal_archive is not None:
            out["schedule"] = self.schedule(transition, gst)
            if transition is not None:
                out["hover_schedule"] = self.schedule(transition, gst, "hover")
        out["goal_stages"] = self.goal_stages
        out["step_rows"], out["step_s"] = self.fstp, self.step_s
        out["box"] = {"min": self.box[0].tolist(), "max": self.box[1].tolist(), "source": self.box[2]}
        return out

    @torch.no_grad()
    def plan(self, image, pose, jaw, goal_index=-1, seed=0):
        if not self.goals:
            raise ValueError("no goal set: POST /goal first")
        goal = self.goals[goal_index]
        pose = np.asarray(pose, dtype=np.float64)
        lo, hi, _ = self.box
        outside = float(np.max(np.maximum(lo - pose[:3], 0) + np.maximum(pose[:3] - hi, 0)))
        rot_err = (Rotation.from_rotvec(pose[3:6]) * Rotation.from_rotvec(RECORDED_ROTVEC).inv()).magnitude()
        if outside > OUTSIDE_TOL_M or rot_err > ROT_TOL_RAD:
            raise ValueError(f"measured pose {np.round(pose, 4).tolist()} is {outside * 1e3:.0f} mm outside the "
                             f"workspace box or {np.rad2deg(rot_err):.1f} deg off the recorded orientation; park the "
                             f"arm at the start pose first")  # fmt: skip
        s7 = hanoi_states(pose[None], np.full((1, 8), float(jaw)))
        z0 = self.encode(np.asarray(image)[None])
        s = torch.as_tensor(s7[None], dtype=torch.float32, device=self.device)
        cem_kw, close = self.cem_kw, None
        if self.goal_stages is not None:  # per-goal options (paper_protocol_eval.aids; none apply by default)
            o = self.options
            axis, close = aids(self.goal_stages[goal_index], o["fix_x"], o["hold_grasp"], o["hold_open"], o["hold_xy"])
            cem_kw = cem_kw | {"axis": axis}
        held = close is not None
        torch.manual_seed(seed)
        with self._autocast():
            plan_fn = self.mpc.cem_categorical if self.sampling == "cat" else self.mpc.cem
            plan = plan_fn(context_frame=z0, context_pose=s, goal_frame=goal, world_model=self.step_fn,
                           close_gripper=close, objective=self.objective, **cem_kw)[0].float()  # fmt: skip
            final = rollout(self.step_fn, z0, s, plan[None]).float()
        a = plan[0].cpu().numpy()  # execute the first planned action, then replan
        # safety, after planning: clamp the target into the box, then cap the executed step at max_step
        step = np.clip(pose[:3] + a[:3], lo, hi) - pose[:3]
        norm = float(np.linalg.norm(step))
        scaled = norm > self.max_step
        if scaled:
            step *= self.max_step / norm
        target = np.concatenate([pose[:3] + step, RECORDED_ROTVEC])
        closedness = float(np.clip(s7[0, 6] + a[6], 0.0, 1.0))
        return {
            "target_pose": target.tolist(),
            "gripper_intent": "close" if closedness > 0.5 else "open",
            "action": a.tolist(),
            "executed_delta": (target[:3] - pose[:3]).tolist(),
            "safety_scaled": bool(scaled),
            "goal_index": goal_index % len(self.goals),
            "seed": seed,
            "gripper_held": bool(held),
            "axis_held": {int(k): float(v) for k, v in cem_kw.get("axis", {}).items()},
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
                self._send(200, {"ok": True, "checkpoint": planner.checkpoint, "goal_archive": planner.goal_archive,
                                 "goal_bundle": planner.goal_bundle, "predictor_chunk": planner.predictor_chunk,
                                 "protocol": planner.protocol, "planner": planner.cem_kw, "options": planner.options,
                                 "planner_preset": planner.planner_preset, "objective": planner.objective.__name__,
                                 "sampling": planner.sampling, "action_bounds": planner.action_bounds,
                                 "max_step_m": planner.max_step, "step_rows": planner.fstp,
                                 "step_s": planner.step_s, "box": {"min": planner.box[0].tolist(),
                                 "max": planner.box[1].tolist(), "source": planner.box[2]}, "goal": planner.goal_desc,
                                 "n_goals": len(planner.goals)})  # fmt: skip
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
                    opts = {k: bool(z[k]) for k in ("fix_x", "hold_grasp", "hold_open", "hold_xy") if k in z.files}
                    if "subgoals" in z.files:
                        opts["subgoals"] = str(z["subgoals"])
                    return self._send(200, planner.set_goal(images=images, board=board, transition=transition, **opts))
                if self.path == "/plan":
                    t0 = time.time()
                    gi = int(z["goal_index"]) if "goal_index" in z.files else -1
                    seed = int(z["seed"]) if "seed" in z.files else 0
                    out = planner.plan(z["image"], z["pose"], float(z["jaw"]), goal_index=gi, seed=seed)
                    out["seconds"] = time.time() - t0
                    return self._send(200, out)
                self._send(404, {"error": "unknown path"})
            except Exception as e:
                traceback.print_exc()
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
    p.add_argument("--goal_bundle", default=None, help="goals.npz from export_robot_bundle (instead of --goal_archive)")
    p.add_argument("--protocol", choices=tuple(PLANNERS), default="plain")
    p.add_argument("--final_stage", type=int, choices=(8, 9), default=8, help="paper protocol: final goal frame")
    p.add_argument("--fix_x", action="store_true",
                   help="deviation, off by default: hold x (play-data models only; never from the episode start pose)")
    p.add_argument("--hold_grasp", action="store_true", help="deviation, off by default: gripper closed during goal 2")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--max_speed", type=float, default=0.10, help="execution safety limit (m/s), after planning")
    p.add_argument("--predictor_chunk", type=int, default=0,
                   help="CEM samples per predictor call (0 = all): same plans, less GPU memory (200 for 16 GB)")
    p.add_argument("--rollout", type=int, default=None, help="deviation: planning horizon (default: the protocol's)")
    p.add_argument("--samples", type=int, default=None, help="deviation: CEM samples (default: the protocol's)")
    p.add_argument("--gripper_std", type=float, default=None, help="deviation: CEM initial gripper std (repo 1.0)")
    p.add_argument("--gripper_clip", type=float, default=None, help="deviation: CEM gripper clip (repo 0.75)")
    p.add_argument("--maxnorm", type=float, default=None,
                   help="deviation: CEM per-axis box (default 0.05 m per fps-4 step, scaled with the model's step)")
    p.add_argument("--subgoals", choices=("paper3", "stages"), default="paper3",
                   help="deviation: stages = one goal per motion stage (default: the paper's three)")
    p.add_argument("--hold_open", action="store_true", help="task aid, off by default: gripper open, descend goal")
    p.add_argument("--hold_xy", action="store_true", help="task aid, off by default: x/y held, descend/grasp goals")
    p.add_argument("--planner", choices=("protocol", "jepawms"), default="protocol",
                   help="deviation: jepawms = the JEPA-WMs V-JEPA 2-AC planner settings (header)")
    p.add_argument("--sampling", choices=("box", "data", "cat"), default="box",
                   help="deviation: data = CEM samples inside the training actions' range; cat = categorical (header)")
    p.add_argument("--guard_floor_mm", type=float, default=None,
                   help="--sampling cat: the arm's low floor in mm (robot_io: workspace transit_z - 25 mm, 166.13 on "
                        "2026-10-13): steps more than 3 mm sideways with an end below it are not planned (the guard "
                        "would abort them; the guard itself stays on)")
    p.add_argument("--box_feasible", action="store_true",
                   help="--sampling cat: do not plan steps that end outside the workspace box (they would be clamped)")
    p.add_argument("--limit_codes_to_cap", action="store_true",
                   help="--sampling cat: drop codes longer than the step cap instead of scaling the executed step")
    args = p.parse_args()
    planner = GoalPlanner(args.fname, args.checkpoint, args.goal_archive, args.max_speed, args.protocol,
                          args.final_stage, args.fix_x, args.hold_grasp, args.goal_bundle,
                          predictor_chunk=args.predictor_chunk,
                          planner_overrides={"rollout": args.rollout, "samples": args.samples,
                                             "gripper_std": args.gripper_std, "gripper_clip": args.gripper_clip,
                                             "maxnorm": args.maxnorm},
                          hold_open=args.hold_open, hold_xy=args.hold_xy, subgoals=args.subgoals,
                          planner=args.planner, sampling=args.sampling, guard_floor_mm=args.guard_floor_mm,
                          limit_codes_to_cap=args.limit_codes_to_cap, box_feasible=args.box_feasible)  # fmt: skip
    serve(planner, args.host, args.port)


if __name__ == "__main__":
    main()

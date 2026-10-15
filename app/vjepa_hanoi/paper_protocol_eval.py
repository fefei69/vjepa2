# One Hanoi game move with the V-JEPA 2-AC paper's pick-and-place protocol, checked offline against the expert.
#
#   python -m app.vjepa_hanoi.paper_protocol_eval --fname <yaml> --checkpoint <best.pt> \
#       --archive <held-out split .npz> --goal_archive <training split .npz> --goals paper3 --switch time --out <json>
#
# The V-JEPA 2 paper (arXiv 2506.09985, sec. 4.2) does pick-and-place with "two sub-goal images ... in addition to the
# final goal. The first goal image shows the object being grasped, the second goal image shows the object in the
# vicinity of the goal position", switching goals on a fixed time schedule. It plans with CEM (800 samples, 10
# refinement steps) at planning horizon 1 on the L1 energy, executing one action before re-planning. For one Hanoi
# ring transfer, the three goal frames are:
#   goal 1 "ring grasped"           end of motion stage 4 (grasp): jaw closed on the ring at the source peg
#   goal 2 "ring near the target"   end of stage 6 (transit): ring above the target peg
#   goal 3 "final"                  end of stage 8 (release: ring on the target peg, arm still at it) by default, or
#                                   with --final_stage 9 the end of stage 9 (arm retreated; the end of the game move)
# --goal_source train (default): each goal is ONE recorded frame of the first complete training move of the same board
# transition. These are experimenter-provided images, as in the paper; no held-out frame is used. --goal_source test:
# the held-out move's own frames (a same-scene goal photo; this is the source the goal-distance sweep used).
# --goals final_only gives the same planner the final image only.
#
# Rows are held-out rows along each move, one model step apart (step k = k x 8 raw rows after the move starts). At each
# row the planner makes one step toward the active goal, and that step is compared with the expert's actual next step
# (rows r -> r + 8, ending no later than the final goal frame). --switch stage is perfect switching, an upper bound:
# the active goal is the first one the expert has not reached by its next sample. --switch time is the paper's fixed
# schedule, with the goal chosen from k alone. --schedule transition (default; the arm server's rule): the switch points
# are the median step at which the training demos of the SAME transition reach goals 1 and 2 (all training moves if the
# transition has none). --schedule global: the median over all training moves, as in the first runs (before 2026-10-08).
#
# Deviations from the paper, stated:
#   - The repo planner (notebooks/utils/mpc_utils.py::cem) clips each axis to a box (maxnorm 0.05 m), where the paper
#     uses an L1 ball of radius 0.075. Neither binds the expert (steps <= 3.3 cm), but CEM's plans sit at the clip
#     in most rows (robot_results/2026-10-13/RESULTS.md).
#   - The paper's 4/10/4-step schedule is specific to its task, so the Hanoi schedule is set from expert timing.
#   - --fix_x (off by default) holds x at 0 via the repo planner's axis option. x never varies in the play data, so
#     a play model cannot learn its effect and plans it at random. In the one-move data x moves forward during the
#     approach only (up to 56 mm per fps-1 step, within 3 mm in later stages), so do not fix it from the start pose.
#   - --hold_grasp (off by default) holds the gripper closed while goal 2 is active, via the repo planner's
#     close_gripper option.
# Further options for the descend/grasp problem found on the arm (robot_results/2026-10-13/RESULTS.md), all off by
# default, each labelled in the output:
#   planner deviations (plain CEM / goal settings, no task knowledge in the planner):
#   - --gripper_std S / --gripper_clip C: CEM's initial gripper std and clip (repo: 1.0 and 0.75)
#   - --goals stages: one goal per motion stage instead of the paper's three: the ends of stages 3 descend, 4 grasp,
#     5 lift, 6 transit, 7 insert and the final stage, from the same training demo, on the same kind of time schedule
#   task aids (outside the paper baseline; they encode what the expert does in a stage):
#   - --hold_open: gripper action held at 0 (no change: stays open) while a stage-3 (descend) goal is active, via the
#     repo planner's axis option; needs --goals stages
#   - --hold_xy: x and y actions held at 0 while a stage-3 or stage-4 goal (descend, grasp; paper goal 1) is active.
#     With --goals paper3, goal 1 also covers the approach from the episode start, so use it from the hover only
#   - in --goals stages mode --hold_grasp holds the gripper closed while the lift, transit and insert goals are active
# Planner presets, off by default (planner deviations, no task knowledge; labelled P in descent_report):
#   - --protocol jepawms: the V-JEPA 2-AC planner of JEPA-WMs (arXiv 2512.24497; jepa-wms DROID vj2ac planning config):
#     CEM 300 samples x 15 iterations, top 10, horizon 3, no momentum, initial std 0.1 on every sampled dim, clip +-0.1
#     on xyz (not scaled with the model step) and +-0.75 on the gripper, the current mean kept as sample 0, no final
#     gripper rounding, L2 energy on the last imagined frame, predictor context a sliding window of the last 2 frames.
#     Not reproduced: they also sample rotation (+-0.1 rad, ~10x any rotation step in this data; here it stays 0),
#     and their DROID evaluation plans once, open loop, with the goal 3 steps ahead.
#   - --sampling data: CEM samples kept inside the range of the training actions at the model step
#     (app/vjepa_hanoi/action_bounds.py on --goal_archive): per-axis signed xyz box at the 0.1/99.9th percentiles,
#     initial xyz std = the per-axis spread of the moving training steps; gripper changes in [-1, 1] (training changes
#     are 0 or about +-1), initial gripper std 0.5, clipped to what the imagined closedness allows; a sample either
#     moves or opens/closes (|gripper| >= 0.5 zeroes its xyz, else its gripper: the expert never does both in one
#     step); the mean is kept as sample 0 and the best-scoring sample is executed instead of the elite mean (which
#     may average a moving and a closing elite into an unseen action). Momentum, horizon and energy: from --protocol.
#   - --sampling cat: categorical planning over a finite action set (mpc_utils.cem_categorical): 'stay', 'close',
#     'open' and the --cat_k k-means centers of the moving training steps at the model step
#     (action_bounds.action_codebook on --goal_archive). Every code sequence is scored when there are at most 4096
#     (exact search: H=1, and H=2 with 51 codes); otherwise categorical CEM (samples / iterations / top-k from
#     --protocol) starting from the codes' training frequencies ** --cat_temp. Gripper feasibility as for data.
#     --cat_max_mm drops the moving codes longer than that (the arm's 36 mm step rule at fps 1; goal_planner does
#     this at its --max_speed step cap). --guard_floor_mm 166.13 (robot_io's low floor) scores a step more than 3 mm
#     sideways with an end below that height as infeasible, so the planner only ranks steps the arm's low-lateral guard
#     executes instead of aborting the run (mpc_utils.cem_categorical low_floor).
# --eval_stages 3,4 evaluates only rows in those motion stages (row selection, not a planner change): the arm runner
# starts at the hover, so the approach rows (stages 1-2) do not apply to it.

import argparse
import itertools
import json
import time
from collections import defaultdict

import h5py
import numpy as np
import torch
import yaml

from app.vjepa_droid.transforms import make_transforms
from app.vjepa_hanoi.ac_step import encode_clips
from app.vjepa_hanoi.eval import STAGES, build_models, strip_prefixes
from app.vjepa_hanoi.goal_distance_eval import summarize
from app.vjepa_hanoi.hanoi import RATE_HZ, hanoi_states, poses_to_diffs, step_rows
from app.vjepa_hanoi.plan_eval import load_mpc_utils, make_step_fn, windowed

FSTP = 8  # raw rows per model step of the fps-4 models (30 Hz); main() uses the config's data.fps
MAXNORM_FPS4 = 0.05  # repo cem per-axis box at fps 4; other strides scale it with the step length (same speed box)
BOARDS = ["".join(b) for b in itertools.product("ABC", repeat=4)]  # board_indices order, ring 1 first
GOAL_STAGES = (4, 6)  # goals 1 and 2: the last row of these motion stages; goal 3 is the end of --final_stage
STAGE_GOALS = (3, 4, 5, 6, 7)  # --goals stages: descend, grasp, lift, transit, insert; then the end of --final_stage


def goal_stages(goals, final_stage):
    """Motion stage whose last row is each goal frame, goals in order."""
    if goals == "stages":
        return STAGE_GOALS + tuple(range(8, final_stage + 1))
    return GOAL_STAGES + (final_stage,)


def aids(stage, fix_x=False, hold_grasp=False, hold_open=False, hold_xy=False):
    """The repo planner's (axis, close_gripper) for the active goal's motion stage under the opted-in deviations.
    axis indexes CEM's sampled [dx, dy, dz, grip] vector."""
    axis = {}
    if fix_x:
        axis[0] = 0.0
    if hold_xy and stage in (3, 4):
        axis.update({0: 0.0, 1: 0.0})
    if hold_open and stage == 3:
        axis[3] = 0.0
    return axis, (0 if hold_grasp and stage in (5, 6, 7) else None)


def moves(archive):
    """Complete game moves (motion stages 1..9 in order) of a split archive, in archive order, de-duplicated."""
    z = np.load(archive)
    obs, fi, seg = z["source_observation_indices"], z["file_indices"], z["segment_bounds"]
    mv, st, board = z["move_indices"], z["motion_stages"], z["board_indices"]
    out, seen, incomplete = [], set(), 0
    for key in np.unique(np.stack([fi, seg[:, 0]], 1), axis=0):
        sel = np.nonzero((fi == key[0]) & (seg[:, 0] == key[1]))[0]
        sel = sel[np.argsort(obs[sel])]
        for m in np.unique(mv[sel]):
            s = sel[mv[sel] == m]
            if not np.isin(st[s], np.arange(1, 10)).any():
                continue  # terminal hold/padding row: not a move
            runs = [int(st[s[0]])] + [int(b) for a, b in zip(st[s][:-1], st[s][1:]) if b != a]
            if runs != list(range(1, 10)):
                incomplete += 1  # cut by a segment boundary
                continue
            if (int(key[0]), int(obs[s[0]])) in seen:
                continue
            seen.add((int(key[0]), int(obs[s[0]])))
            # the board label switches at the release -> retreat boundary, so the first and last rows give the move
            transition = f"{BOARDS[board[s[0]]]}>{BOARDS[board[s[-1]]]}"
            out.append({"file": int(key[0]), "transition": transition, "rows": obs[s], "stages": st[s]})
    return out, [str(p) for p in z["source_paths"]], incomplete


def goal_rows(move, final_stage, stages=None):
    """Raw rows of goals 1..n in a complete move: the last usable row of each goal stage (default: 4, 6, final)."""
    stages = stages or GOAL_STAGES + (final_stage,)
    return {g: int(move["rows"][move["stages"] == s].max()) for g, s in enumerate(stages, 1)}


def time_schedule(train_moves, final_stage, stages=None, fstp=FSTP, start="move"):
    """Median step at which the expert reaches goals 1..n, on the training split, counted from the move's first row
    (start "move") or from the end of its approach, the hover above the source peg (start "hover": each demo's last
    stage-2 row, for runs that start at the hover)."""
    n = len(stages or GOAL_STAGES + (final_stage,))

    def origin(m):
        return int(m["rows"][0]) if start == "move" else int(m["rows"][m["stages"] == 2].max())

    reach = np.array([[(goal_rows(m, final_stage, stages)[g] - origin(m)) / fstp for g in range(1, n + 1)]
                      for m in train_moves])  # fmt: skip
    return [int(round(x)) for x in np.median(reach, axis=0)]


def l2(a, b):
    return torch.mean((a - b) ** 2, dim=-1)


JEPAWMS = dict(rollout=3, samples=300, cem_steps=15, topk=10, momentum_mean=0.0, momentum_std=0.0,
               momentum_mean_gripper=0.0, momentum_std_gripper=0.0, maxnorm=0.1, gripper_std=0.1, gripper_clip=0.75,
               include_mean=True, round_gripper=0.0)  # fmt: skip
JEPAWMS_WINDOW = 2
GRIPPER_STD_DATA, EXCLUSIVE_GRIPPER = 0.5, 0.5  # --sampling data: P(|g| >= 0.5) = 0.32 of the first samples


def data_sampling(bounds):
    """cem kwargs of --sampling data from action_bounds.action_bounds(train split, fps)."""
    return dict(init_std=bounds["std"] + [GRIPPER_STD_DATA], lo=bounds["lo"] + [-1.0], hi=bounds["hi"] + [1.0],
                include_mean=True, exclusive_gripper=EXCLUSIVE_GRIPPER, gripper_feasible=True, return_best=True)  # fmt: skip


def default_maxnorm(fstp):
    """The repo's 0.05 m box per fps-4 step, scaled to the step length (the same speed box, 187.5 mm/s)."""
    return MAXNORM_FPS4 if fstp == FSTP else MAXNORM_FPS4 * fstp / FSTP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fname", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--archive", required=True, help="held-out split (test)")
    p.add_argument("--goal_archive", required=True, help="training split: goal frames (train source) and schedule")
    p.add_argument("--goals", choices=("paper3", "final_only", "stages"), default="paper3")
    p.add_argument("--switch", choices=("stage", "time"), default="time")
    p.add_argument("--schedule", choices=("transition", "global"), default="transition", help="time-switch medians")
    p.add_argument("--goal_source", choices=("train", "test"), default="train")
    p.add_argument("--final_stage", type=int, choices=(8, 9), default=8)
    p.add_argument("--max_rows", type=int, default=600)
    p.add_argument("--row_stride", type=int, default=None,
                   help="raw rows between evaluated rows (default: one model step); each row still plans one model "
                        "step, e.g. 8 to evaluate an fps-1 model as densely as an fps-4 one")
    p.add_argument("--samples", type=int, default=800, help="paper: 800")
    p.add_argument("--cem_steps", type=int, default=10, help="paper: 10 refinement steps")
    p.add_argument("--rollout", type=int, default=1, help="paper: planning horizon 1")
    p.add_argument("--maxnorm", type=float, default=None,
                   help="repo cem per-axis box (default: 0.05 per fps-4 step, scaled with the step; paper: L1 0.075)")
    p.add_argument("--fix_x", action="store_true", help="deviation, off by default: hold x (never varies in data)")
    p.add_argument("--hold_grasp", action="store_true", help="deviation, off by default: gripper closed during goal 2")
    p.add_argument("--gripper_std", type=float, default=1.0, help="planner deviation: CEM initial gripper std (1.0)")
    p.add_argument("--gripper_clip", type=float, default=0.75, help="planner deviation: CEM gripper clip (0.75)")
    p.add_argument("--hold_open", action="store_true", help="task aid, off by default: gripper open (descend goal)")
    p.add_argument("--hold_xy", action="store_true", help="task aid, off by default: x/y held (descend/grasp goals)")
    p.add_argument("--protocol", choices=("paper", "jepawms"), default="paper",
                   help="planner preset: paper (V-JEPA 2 / repo cem) or jepawms (see the header)")
    p.add_argument("--sampling", choices=("box", "data", "cat"), default="box",
                   help="box: the repo's clip box; data: keep samples inside the training actions' range; "
                        "cat: categorical over a codebook of training actions (header)")
    p.add_argument("--cat_k", type=int, default=48, help="--sampling cat: k-means codes of the moving steps")
    p.add_argument("--cat_temp", type=float, default=0.5,
                   help="--sampling cat, categorical CEM: initial distribution = code frequency ** temp (0 uniform)")
    p.add_argument("--cat_max_mm", type=float, default=None,
                   help="--sampling cat: drop moving codes longer than this (as goal_planner does at its step cap)")
    p.add_argument("--guard_floor_mm", type=float, default=None,
                   help="--sampling cat: rank only steps the arm's low-lateral guard executes (header)")
    p.add_argument("--box_feasible", action="store_true",
                   help="--sampling cat: rank only steps ending inside goal_planner's workspace box")
    p.add_argument("--eval_stages", default="", help="evaluate only rows in these motion stages, e.g. 3,4")
    p.add_argument("--predictor_chunk", type=int, default=0, help="CEM samples per predictor call (memory only)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.fname) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    d = cfg["data"]
    fstp = step_rows(d.get("fps", 4))  # raw rows per model step (8 at fps 4)
    maxnorm = default_maxnorm(fstp) if args.maxnorm is None else args.maxnorm
    row_stride = args.row_stride or fstp
    crop = d.get("crop_size", 256)
    tpf = (crop // d["patch_size"]) ** 2
    norm = cfg["loss"]["normalize_reps"]
    dtype = torch.bfloat16 if str(cfg["meta"].get("dtype")).lower() == "bfloat16" else torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder, predictor = build_models(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    encoder.load_state_dict(strip_prefixes(ckpt["target_encoder"]), strict=True)
    predictor.load_state_dict(strip_prefixes(ckpt["predictor"]), strict=True)
    del ckpt
    encoder.eval()
    predictor.eval()
    mpc = load_mpc_utils()
    step_fn = make_step_fn(predictor, tpf, norm, mpc.compute_new_pose)
    if args.protocol == "jepawms":
        step_fn = windowed(step_fn, JEPAWMS_WINDOW)
    if args.predictor_chunk:
        from app.vjepa_hanoi.goal_planner import chunked  # goal_planner imports this module

        step_fn = chunked(step_fn, args.predictor_chunk)
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop,
    )
    cem_kw = dict(rollout=args.rollout, samples=args.samples, topk=10, cem_steps=args.cem_steps,
                  momentum_mean=0.15, momentum_std=0.15, maxnorm=maxnorm)  # fmt: skip
    if (args.gripper_std, args.gripper_clip) != (1.0, 0.75):  # only when set, so default runs stay byte-identical
        cem_kw |= {"gripper_std": args.gripper_std, "gripper_clip": args.gripper_clip}
    objective, bounds, codebook = mpc.l1, None, None
    if args.protocol == "jepawms":
        if (args.rollout, args.samples, args.cem_steps, args.maxnorm, args.gripper_std, args.gripper_clip) != (
            1, 800, 10, None, 1.0, 0.75):  # fmt: skip
            p.error("--protocol jepawms fixes horizon, samples, iterations, box and gripper std/clip")
        cem_kw, objective = dict(JEPAWMS), l2
    if args.sampling == "data":
        from app.vjepa_hanoi.action_bounds import action_bounds

        bounds = action_bounds(args.goal_archive, d.get("fps", 4))
        cem_kw |= data_sampling(bounds)
    plan_fn = mpc.cem
    if args.sampling == "cat":
        from app.vjepa_hanoi.action_bounds import action_codebook, limit_codebook

        codebook = action_codebook(args.goal_archive, d.get("fps", 4), k_move=args.cat_k)
        if args.cat_max_mm is not None:
            codebook = limit_codebook(codebook, args.cat_max_mm / 1e3)
            print(f"codebook limited to {len(codebook['codes'])} codes of at most {args.cat_max_mm:g} mm")
        cem_kw = {k: cem_kw[k] for k in ("rollout", "samples", "cem_steps", "topk")} | {
            "codebook": codebook["codes"], "code_prior": codebook["prior"], "prior_temp": args.cat_temp}
        plan_fn = mpc.cem_categorical
    if args.guard_floor_mm is not None:
        if args.sampling != "cat":
            p.error("--guard_floor_mm is a --sampling cat option")
        cem_kw["low_floor"] = args.guard_floor_mm / 1e3
    if args.box_feasible:
        if args.sampling != "cat":
            p.error("--box_feasible is a --sampling cat option")
        xyz = np.load(args.goal_archive)["measured_xyz"]  # goal_planner's box: the recorded range, padded
        cem_kw["box"] = ((xyz.min(0) - [0.005, 0.005, 0.001]).tolist(), (xyz.max(0) + 0.005).tolist())
    if args.hold_open and args.goals != "stages":
        p.error("--hold_open acts on the descend goal: use --goals stages")
    gstages = goal_stages(args.goals, args.final_stage)
    n_goals = len(gstages)
    eval_stages = {int(x) for x in args.eval_stages.split(",") if x}

    train_moves, train_paths, train_incomplete = moves(args.goal_archive)
    test_moves, test_paths, test_incomplete = moves(args.archive)
    reach = time_schedule(train_moves, args.final_stage, gstages, fstp)  # steps at which the expert reaches each goal
    reach_by = {}  # transition -> its own medians (the arm server's rule)
    for t in {m["transition"] for m in train_moves}:
        reach_by[t] = time_schedule([m for m in train_moves if m["transition"] == t], args.final_stage, gstages, fstp)
    exemplar = {}  # transition -> (file, goal rows) of its first complete training move
    for m in train_moves:
        exemplar.setdefault(m["transition"], (m["file"], goal_rows(m, args.final_stage, gstages)))
    files = {"train": [h5py.File(pth, "r") for pth in train_paths], "test": [h5py.File(pth, "r") for pth in test_paths]}

    # rows: --row_stride raw rows apart (default one model step) from the start of each held-out move; the expert's
    # next sample (one model step later) must be usable and must not pass the final goal frame
    cands, skipped_no_exemplar = [], 0
    for mi, m in enumerate(test_moves):
        if m["transition"] not in exemplar:
            skipped_no_exemplar += 1
            continue
        usable, g_test = set(m["rows"].tolist()), goal_rows(m, args.final_stage, gstages)
        r0 = int(m["rows"][0])
        for r in range(r0, g_test[n_goals] - fstp + 1, row_stride):
            k = (r - r0) // fstp if row_stride == fstp else (r - r0) / fstp  # model steps since the move's start
            if r not in usable or r + fstp not in usable:
                continue
            if eval_stages and int(m["stages"][m["rows"] == r][0]) not in eval_stages:
                continue
            g_stage = next(g for g in range(1, n_goals + 1) if g_test[g] >= r + fstp)
            rt = reach_by.get(m["transition"], reach) if args.schedule == "transition" else reach
            g_time = next((g for g in range(1, n_goals) if k < rt[g - 1]), n_goals)
            g = n_goals if args.goals == "final_only" else (g_stage if args.switch == "stage" else g_time)
            cands.append({"move": mi, "row": r, "k": k, "goal": g, "goal_stage_switch": g_stage})
    pick = np.linspace(0, len(cands) - 1, min(args.max_rows, len(cands))).astype(int)

    @torch.no_grad()
    def encode(image):
        clip = transform(np.asarray(image)[None])[None].to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            return encode_clips(encoder, clip, norm).float().view(1, 1, tpf, -1)

    goal_cache = {}
    rec = defaultdict(list)
    t0 = time.time()
    for n, i in enumerate(pick):
        c = cands[i]
        m, r, g = test_moves[c["move"]], c["row"], c["goal"]
        h = files["test"][m["file"]]
        g_test = goal_rows(m, args.final_stage, gstages)
        if args.goal_source == "train":
            gsrc, gfile, grow = "train", exemplar[m["transition"]][0], exemplar[m["transition"]][1][g]
        else:
            gsrc, gfile, grow = "test", m["file"], g_test[g]
        if (gsrc, gfile, grow) not in goal_cache:
            goal_cache[(gsrc, gfile, grow)] = encode(files[gsrc][gfile]["pixels"][grow])
        pose = h["state"][[r, r + fstp]].astype(np.float64)
        jaw = h["proprio"][[r, r + fstp], 6]
        s7 = hanoi_states(pose, np.repeat(jaw[:, None], 8, axis=1))
        true = poses_to_diffs(s7)[0]
        z0 = encode(h["pixels"][r])
        pose_t = torch.as_tensor(s7[None, :1], dtype=torch.float32, device=device)
        torch.manual_seed(n)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            axis, close = aids(gstages[g - 1], args.fix_x, args.hold_grasp and args.goals != "final_only",
                               args.hold_open, args.hold_xy)  # fmt: skip
            plan = plan_fn(context_frame=z0, context_pose=pose_t, goal_frame=goal_cache[(gsrc, gfile, grow)],
                           world_model=step_fn, close_gripper=close, objective=objective,
                           **(cem_kw | {"axis": axis}))[0]  # fmt: skip
        rec["true"].append(true)
        rec["plan"].append(plan[0].float().cpu().numpy())
        rec["stage"].append(int(m["stages"][m["rows"] == r][0]))
        rec["z"].append(float(pose[0, 2]))
        rec["closedness"].append(float(s7[0, 6]))
        rec["goal_stage"].append(gstages[g - 1])
        rec["goal"].append(g)
        rec["goal_stage_switch"].append(c["goal_stage_switch"])
        rec["goal_seconds"].append((g_test[g] - r) / RATE_HZ)  # when the expert reaches the active goal (< 0: passed)
        rec["step_in_move"].append(c["k"])
        rec["file_index"].append(m["file"])
        rec["row"].append(r)
        rec["goal_file_index"].append(gfile)
        rec["goal_row"].append(grow)
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(pick)} rows, {(time.time() - t0) / (n + 1):.2f} s/row", flush=True)
    rec = {key: np.asarray(v) for key, v in rec.items()}
    np.savez_compressed(args.out.replace(".json", "_rows.npz"), **rec)

    allm = np.ones(len(rec["stage"]), bool)
    result = {
        "goals": args.goals,
        "switch": args.switch if args.goals != "final_only" else "none",
        "goal_source": args.goal_source,
        "final_stage": args.final_stage,
        "fix_x": args.fix_x,
        "hold_grasp": args.hold_grasp,
        "hold_open": args.hold_open,
        "hold_xy": args.hold_xy,
        "eval_stages": sorted(eval_stages) or "all",
        "goal_frames": {str(g): f"end of stage {s} ({STAGES[s]})" for g, s in enumerate(gstages, 1)},
        "schedule": args.schedule,
        "time_schedule_reach_steps": reach,
        "time_schedule_reach_steps_by_transition": reach_by if args.schedule == "transition" else None,
        "time_schedule_steps_per_goal": np.diff([0] + reach).tolist(),
        "step_rows": fstp,
        "row_stride": row_stride,
        "step_s": fstp / RATE_HZ,
        "planner": cem_kw | {"source": "notebooks/utils/mpc_utils.py::cem; paper: 800 samples, 10 steps, horizon 1",
                             "objective": objective.__name__,
                             "context_window": JEPAWMS_WINDOW if args.protocol == "jepawms" else "all"},
        "protocol": args.protocol,
        "sampling": args.sampling,
        "action_bounds": bounds,
        "codebook": codebook,
        "checkpoint": args.checkpoint,
        "archive": args.archive,
        "goal_archive": args.goal_archive,
        "moves": {"test": len(test_moves), "test_incomplete_skipped": test_incomplete,
                  "test_without_training_exemplar": skipped_no_exemplar, "train": len(train_moves),
                  "train_incomplete_skipped": train_incomplete, "candidate_rows": len(cands)},  # fmt: skip
        "switch_agreement_with_stage": float(np.mean(rec["goal"] == rec["goal_stage_switch"])),
        "overall": summarize(rec, allm),
        "by_stage": {STAGES[int(s)]: summarize(rec, rec["stage"] == s) for s in np.unique(rec["stage"])},
        "by_goal": {f"goal_{int(g)}": summarize(rec, rec["goal"] == g) for g in np.unique(rec["goal"])},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    o = result["overall"]
    by = " ".join(f"{s.split('_')[0]}={v['direction_correct']:.2f}" for s, v in result["by_stage"].items())
    print(
        f"[{args.goals} switch={result['switch']} source={args.goal_source} final=stage{args.final_stage} "
        f"schedule={result['time_schedule_steps_per_goal']}] {o['rows']} rows: direction correct "
        f"{o['direction_correct']:.2f}, within 45 deg {o['direction_within_45deg']:.2f}, gripper "
        f"{o['gripper_sign_accuracy']:.2f} ({o['gripper_events']}) | by stage: {by} -> {args.out}"
    )


if __name__ == "__main__":
    main()

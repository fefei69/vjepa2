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
#     uses an L1 ball of radius 0.075. Neither binds here: expert steps are <= 3.3 cm.
#   - The paper's 4/10/4-step schedule is specific to its task, so the Hanoi schedule is set from expert timing.
#   - --fix_x (off by default) holds x at 0 via the repo planner's axis option. x never varies in this data, so
#     the model cannot learn its effect, and the planned x is otherwise random.
#   - --hold_grasp (off by default) holds the gripper closed while goal 2 is active, via the repo planner's
#     close_gripper option.

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
from app.vjepa_hanoi.hanoi import hanoi_states, poses_to_diffs
from app.vjepa_hanoi.plan_eval import load_mpc_utils, make_step_fn

FSTP = 8  # raw rows per model step (30 Hz, fps 4)
BOARDS = ["".join(b) for b in itertools.product("ABC", repeat=4)]  # board_indices order, ring 1 first
GOAL_STAGES = (4, 6)  # goals 1 and 2: the last row of these motion stages; goal 3 is the end of --final_stage


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


def goal_rows(move, final_stage):
    """Raw rows of goals 1, 2, 3 in a complete move: the last usable row of stages 4, 6 and final_stage."""
    return {g: int(move["rows"][move["stages"] == s].max()) for g, s in zip((1, 2, 3), GOAL_STAGES + (final_stage,))}


def time_schedule(train_moves, final_stage):
    """Median step (from the move's first row) at which the expert reaches goals 1, 2 and 3, on the training split."""
    reach = np.array([[(goal_rows(m, final_stage)[g] - m["rows"][0]) / FSTP for g in (1, 2, 3)] for m in train_moves])
    return [int(round(x)) for x in np.median(reach, axis=0)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fname", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--archive", required=True, help="held-out split (test)")
    p.add_argument("--goal_archive", required=True, help="training split: goal frames (train source) and schedule")
    p.add_argument("--goals", choices=("paper3", "final_only"), default="paper3")
    p.add_argument("--switch", choices=("stage", "time"), default="time")
    p.add_argument("--schedule", choices=("transition", "global"), default="transition", help="time-switch medians")
    p.add_argument("--goal_source", choices=("train", "test"), default="train")
    p.add_argument("--final_stage", type=int, choices=(8, 9), default=8)
    p.add_argument("--max_rows", type=int, default=600)
    p.add_argument("--samples", type=int, default=800, help="paper: 800")
    p.add_argument("--cem_steps", type=int, default=10, help="paper: 10 refinement steps")
    p.add_argument("--rollout", type=int, default=1, help="paper: planning horizon 1")
    p.add_argument("--maxnorm", type=float, default=0.05, help="repo cem per-axis box (paper: L1 ball 0.075)")
    p.add_argument("--fix_x", action="store_true", help="deviation, off by default: hold x (never varies in data)")
    p.add_argument("--hold_grasp", action="store_true", help="deviation, off by default: gripper closed during goal 2")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.fname) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    d = cfg["data"]
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
                  momentum_mean=0.15, momentum_std=0.15, maxnorm=args.maxnorm)  # fmt: skip
    if args.fix_x:
        cem_kw["axis"] = {0: 0.0}

    train_moves, train_paths, train_incomplete = moves(args.goal_archive)
    test_moves, test_paths, test_incomplete = moves(args.archive)
    reach = time_schedule(train_moves, args.final_stage)  # steps at which the expert reaches goals 1, 2, 3 (global)
    reach_by = {}  # transition -> its own medians (the arm server's rule)
    for t in {m["transition"] for m in train_moves}:
        reach_by[t] = time_schedule([m for m in train_moves if m["transition"] == t], args.final_stage)
    exemplar = {}  # transition -> (file, goal rows) of its first complete training move
    for m in train_moves:
        exemplar.setdefault(m["transition"], (m["file"], goal_rows(m, args.final_stage)))
    files = {"train": [h5py.File(pth, "r") for pth in train_paths], "test": [h5py.File(pth, "r") for pth in test_paths]}

    # rows: one model step apart from the start of each held-out move; the expert's next sample must be usable and
    # must not pass the final goal frame
    cands, skipped_no_exemplar = [], 0
    for mi, m in enumerate(test_moves):
        if m["transition"] not in exemplar:
            skipped_no_exemplar += 1
            continue
        usable, g_test = set(m["rows"].tolist()), goal_rows(m, args.final_stage)
        r0 = int(m["rows"][0])
        for k in range((g_test[3] - FSTP - r0) // FSTP + 1):
            r = r0 + FSTP * k
            if r not in usable or r + FSTP not in usable:
                continue
            g_stage = next(g for g in (1, 2, 3) if g_test[g] >= r + FSTP)
            rt = reach_by.get(m["transition"], reach) if args.schedule == "transition" else reach
            g_time = 1 if k < rt[0] else (2 if k < rt[1] else 3)
            g = 3 if args.goals == "final_only" else (g_stage if args.switch == "stage" else g_time)
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
        g_test = goal_rows(m, args.final_stage)
        if args.goal_source == "train":
            gsrc, gfile, grow = "train", exemplar[m["transition"]][0], exemplar[m["transition"]][1][g]
        else:
            gsrc, gfile, grow = "test", m["file"], g_test[g]
        if (gsrc, gfile, grow) not in goal_cache:
            goal_cache[(gsrc, gfile, grow)] = encode(files[gsrc][gfile]["pixels"][grow])
        pose = h["state"][[r, r + FSTP]].astype(np.float64)
        jaw = h["proprio"][[r, r + FSTP], 6]
        s7 = hanoi_states(pose, np.repeat(jaw[:, None], 8, axis=1))
        true = poses_to_diffs(s7)[0]
        z0 = encode(h["pixels"][r])
        pose_t = torch.as_tensor(s7[None, :1], dtype=torch.float32, device=device)
        torch.manual_seed(n)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            held = args.hold_grasp and args.goals == "paper3" and g == 2  # carrying toward goal 2
            plan = mpc.cem(context_frame=z0, context_pose=pose_t, goal_frame=goal_cache[(gsrc, gfile, grow)],
                           world_model=step_fn, close_gripper=0 if held else None, **cem_kw)[0]  # fmt: skip
        rec["true"].append(true)
        rec["plan"].append(plan[0].float().cpu().numpy())
        rec["stage"].append(int(m["stages"][m["rows"] == r][0]))
        rec["goal"].append(g)
        rec["goal_stage_switch"].append(c["goal_stage_switch"])
        rec["goal_seconds"].append((g_test[g] - r) / 30.0)  # when the expert reaches the active goal (< 0: passed)
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
        "switch": args.switch if args.goals == "paper3" else "none",
        "goal_source": args.goal_source,
        "final_stage": args.final_stage,
        "fix_x": args.fix_x,
        "hold_grasp": args.hold_grasp,
        "goal_frames": {"1": "end of stage 4 (grasp)", "2": "end of stage 6 (transit)",
                        "3": f"end of stage {args.final_stage} ({STAGES[args.final_stage]})"},  # fmt: skip
        "schedule": args.schedule,
        "time_schedule_reach_steps": reach,
        "time_schedule_reach_steps_by_transition": reach_by if args.schedule == "transition" else None,
        "time_schedule_steps_per_goal": [reach[0], reach[1] - reach[0], reach[2] - reach[1]],
        "planner": cem_kw | {"source": "notebooks/utils/mpc_utils.py::cem; paper: 800 samples, 10 steps, horizon 1"},
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

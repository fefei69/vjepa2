# Closed-loop descend + grasp + lift on recorded frames (no robot, no simulator), from the hover of each held-out
# first move:
#
#   python descent_replay.py --fname <config> --checkpoint <best.pt> --settings cat_stages_h1,stages_h1 --out <json>
#
# The planner runs as on the arm (goal_planner): it starts at the hover above the source peg (the last stage-2 row of
# the held-out demo), plans one model step from the current image and pose, the step is executed, and it replans. The
# goal frames and their time schedule are the arm server's (the first complete training demo of the same transition;
# the per-transition medians counted from the hover). "Executed" here means: the pose is moved by the planned xyz
# step (the gripper becomes closed or open as the arm's gripper_intent, closedness > 0.5), and the next image is the
# held-out demo's recorded frame (30 Hz, every raw row of stages 2-6) nearest to the new tool position among the frames
# with the same gripper state. So the planner sees the consequence of its own steps as long as it stays on the expert's
# path; a case stops (and is reported) when that is no longer true:
#   off path        nearest recorded frame > --max_snap_mm away (e.g. moved sideways near the rods)
#   closed early    closed more than 15 mm above the grasp height (there is no recorded frame of that)
#   dropped         opened again after a valid grasp
# Success: closed within 10 mm of the grasp height and 5 mm of the grasp position laterally, then lifted 30 mm with the
# gripper closed. "grasp height/position" is the tool position at the demo's first closed row. Below the grasp height
# the arm's box clamps (lowest recorded height - 1 mm); here the overshoot is reported (lowest planned height).
# Settings (SETTINGS below; labels as in offline.sbatch): base_h1 (the paper protocol, B), cat_h1, stages_h1,
# cat_stages_h1, cat36_stages_h1, data_stages_h1 (P), and with aids (A, paper_protocol_eval.aids per goal stage):
# *_fixx_* (x held at 0: from the hover x never has to move), aid_*_xy_* (hold_xy + hold_grasp).
import argparse
import json
import time

import h5py
import numpy as np
import torch
import yaml

from app.vjepa_droid.transforms import make_transforms
from app.vjepa_hanoi.ac_step import encode_clips
from app.vjepa_hanoi.action_bounds import action_bounds, action_codebook, limit_codebook
from app.vjepa_hanoi.eval import build_models, strip_prefixes
from app.vjepa_hanoi.hanoi import hanoi_states, step_rows
from app.vjepa_hanoi.paper_protocol_eval import (aids, data_sampling, default_maxnorm, goal_rows, goal_stages, moves,
                                                 time_schedule)  # fmt: skip
from app.vjepa_hanoi.plan_eval import load_mpc_utils, make_step_fn

K = "/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/data/expert6_first_move"
OFF = {"fix_x": False, "hold_xy": False, "hold_grasp": False, "hold_open": False}
SETTINGS = {  # name -> (goals, sampling, cat_max_mm, aids); labels as offline.sbatch: aids (fix_x, hold_*) = A
    "base_h1": ("paper3", "box", None, OFF),
    "cat_h1": ("paper3", "cat", None, OFF),
    "stages_h1": ("stages", "box", None, OFF),
    "cat_stages_h1": ("stages", "cat", None, OFF),
    "cat36_stages_h1": ("stages", "cat", 36.0, OFF),
    "data_stages_h1": ("stages", "data", None, OFF),
    "cat_stages_fixx_h1": ("stages", "cat", None, OFF | {"fix_x": True}),  # x held at 0 (A); from the hover
    "cat36_stages_fixx_h1": ("stages", "cat", 36.0, OFF | {"fix_x": True}),  # x never has to move
    "aid_cat_stages_xy_h1": ("stages", "cat", None, OFF | {"hold_xy": True, "hold_grasp": True}),
    "aid_cat36_stages_xy_h1": ("stages", "cat", 36.0, OFF | {"hold_xy": True, "hold_grasp": True}),
}


def replay_case(m, h, fstp, reach, gstages, plan_step, max_steps=3.0, max_snap_mm=15.0):
    """One held-out move from its hover (header). plan_step(raw row of the shown frame, 7-D pose, goal index) -> the
    planned 7-D action."""
    rows = m["rows"][(m["stages"] >= 2) & (m["stages"] <= 6)]
    rows = np.arange(rows.min(), rows.max() + 1)  # every raw row of the approach .. transit
    st = hanoi_states(h["state"][rows].astype(np.float64), np.repeat(h["proprio"][rows, 6][:, None], 8, 1))
    closed_rec = st[:, 6] > 0.5
    grasp_xyz = st[int(np.argmax(closed_rec)), :3]  # the demo's first closed row
    hover_row = int(m["rows"][m["stages"] == 2].max())
    expert_steps = (int(m["rows"][m["stages"] == 5].max()) - hover_row) / fstp
    i = int(np.nonzero(rows == hover_row)[0][0])
    xyz, closed = st[i, :3].copy(), bool(closed_rec[i])
    n_goals = len(gstages)
    log, outcome, grasped, lowest, close_at = [], None, False, xyz[2], None
    for k in range(int(np.ceil(max_steps * expert_steps))):
        cand = np.nonzero(closed_rec == closed)[0]
        dist = np.linalg.norm(st[cand, :3] - xyz, axis=1)
        j = int(cand[np.argmin(dist)])
        snap = float(dist.min())
        if snap * 1e3 > max_snap_mm:
            outcome = "off path"
            break
        g = next((g for g in range(1, n_goals) if k < reach[g - 1]), n_goals)
        s7 = st[j].copy()
        s7[:3], s7[6] = xyz, float(closed)
        torch.manual_seed(k)
        a = np.asarray(plan_step(int(rows[j]), s7, g), dtype=np.float64)
        now_closed = float(np.clip(float(closed) + a[6], 0.0, 1.0)) > 0.5
        dzg = (s7[2] - grasp_xyz[2]) * 1e3
        latg = float(np.linalg.norm(s7[:2] - grasp_xyz[:2])) * 1e3
        log.append({"k": k, "goal": g, "goal_stage": gstages[g - 1], "row": int(rows[j]), "snap_mm": round(snap * 1e3, 1),
                    "z_mm": round(float(s7[2]) * 1e3, 1), "dz_grasp_mm": round(float(dzg), 1),
                    "lat_grasp_mm": round(latg, 1), "plan_mm": [round(float(x) * 1e3, 1) for x in a[:3]],
                    "grip": round(float(a[6]), 2), "closed": now_closed})  # fmt: skip
        if now_closed and not closed:  # the close command, at the pose before this step's move
            close_at = {"k": k, "dz_grasp_mm": round(float(dzg), 1), "lat_grasp_mm": round(latg, 1)}
            if dzg > 15:
                outcome = "closed early"
                break
            grasped = abs(dzg) <= 10 and latg <= 5
        if closed and not now_closed:
            outcome = "dropped" if grasped else "reopened"
            break
        xyz = xyz + a[:3]
        lowest = min(lowest, xyz[2])
        closed = now_closed
        if closed and grasped and xyz[2] - grasp_xyz[2] >= 0.030:
            outcome = "success"
            break
    return {"outcome": outcome or "timeout", "steps": len(log), "expert_steps_hover_to_lift_end": expert_steps,
            "close": close_at, "lowest_below_grasp_mm": round(float(lowest - grasp_xyz[2]) * 1e3, 1), "log": log}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fname", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--archive", default=f"{K}/test.npz")
    p.add_argument("--goal_archive", default=f"{K}/train.npz")
    p.add_argument("--settings", default="base_h1,cat_stages_h1")
    p.add_argument("--max_snap_mm", type=float, default=15.0)
    p.add_argument("--max_steps", type=float, default=3.0, help="x the expert's steps from the hover to the lift end")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.fname) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    d = cfg["data"]
    fstp = step_rows(d.get("fps", 4))
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
    transform = make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0),
                                random_resize_scale=(1.0, 1.0), reprob=0.0, auto_augment=False, motion_shift=False,
                                crop_size=crop)  # fmt: skip

    @torch.no_grad()
    def encode(image):
        clip = transform(np.asarray(image)[None])[None].to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            return encode_clips(encoder, clip, norm).float().view(1, 1, tpf, -1)

    train_moves, train_paths, _ = moves(args.goal_archive)
    test_moves, test_paths, _ = moves(args.archive)
    codebook = bounds = None
    results = {"checkpoint": args.checkpoint, "fname": args.fname, "step_rows": fstp, "settings": {}}
    for name in args.settings.split(","):
        goals, sampling, cat_max_mm, aid = SETTINGS[name]
        gstages = goal_stages(goals, 8)
        n_goals = len(gstages)
        cem_kw = dict(rollout=1, samples=800, topk=10, cem_steps=10, momentum_mean=0.15, momentum_std=0.15,
                      maxnorm=default_maxnorm(fstp))  # fmt: skip
        plan_fn = mpc.cem
        if sampling == "data":
            bounds = bounds or action_bounds(args.goal_archive, d.get("fps", 4))
            cem_kw |= data_sampling(bounds)
        if sampling == "cat":
            codebook = codebook or action_codebook(args.goal_archive, d.get("fps", 4))
            cb = codebook if cat_max_mm is None else limit_codebook(codebook, cat_max_mm / 1e3)
            cem_kw = {k: cem_kw[k] for k in ("rollout", "samples", "cem_steps", "topk")} | {
                "codebook": cb["codes"], "code_prior": cb["prior"], "prior_temp": 0.5}
            plan_fn = mpc.cem_categorical
        cases = []
        t0 = time.time()
        for m in test_moves:
            same = [t for t in train_moves if t["transition"] == m["transition"]]
            if not same:
                continue
            ex = same[0]  # goal frames: the first complete training demo of this transition (goal_planner's choice)
            ex_rows = goal_rows(ex, 8, gstages)
            reach = time_schedule(same, 8, gstages, fstp, "hover")
            with h5py.File(train_paths[ex["file"]], "r") as h:
                goal_z = [encode(h["pixels"][ex_rows[g]]) for g in range(1, n_goals + 1)]
            h = h5py.File(test_paths[m["file"]], "r")

            def plan_step(row, s7, g):
                axis, close = aids(gstages[g - 1], aid["fix_x"], aid["hold_grasp"], aid["hold_open"], aid["hold_xy"])
                z0 = encode(h["pixels"][row])
                pose = torch.as_tensor(s7[None, None], dtype=torch.float32, device=device)
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype,
                                                     enabled=dtype != torch.float32):  # fmt: skip
                    return plan_fn(context_frame=z0, context_pose=pose, goal_frame=goal_z[g - 1], world_model=step_fn,
                                   close_gripper=close, objective=mpc.l1, **(cem_kw | {"axis": axis})
                                   )[0][0].float().cpu().numpy()  # fmt: skip

            case = replay_case(m, h, fstp, reach, gstages, plan_step, args.max_steps, args.max_snap_mm)
            h.close()
            cases.append({"transition": m["transition"]} | case)
            print(f"{name} {m['transition']}: {case['outcome']} in {case['steps']} steps "
                  f"(expert {case['expert_steps_hover_to_lift_end']:.1f}), close {case['close']}, "
                  f"lowest {case['lowest_below_grasp_mm']:+.1f} mm vs grasp", flush=True)  # fmt: skip
        n_ok = sum(c["outcome"] == "success" for c in cases)
        print(f"== {name}: success {n_ok}/{len(cases)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        results["settings"][name] = {"goals": goals, "sampling": sampling, "cat_max_mm": cat_max_mm, "aids": aid,
                                     "success": n_ok, "cases": cases}  # fmt: skip
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

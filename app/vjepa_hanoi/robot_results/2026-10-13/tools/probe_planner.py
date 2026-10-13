# Offline probes of the planner on single frames (no robot), as run on 2026-10-13 for AAAA>BAAA.
#
#   --mode horizon  goal 1 of the bundle ("ring grasped"), from the live hover frame of a hardware run and from the
#                   held-out demo at the hover and N steps into the descent, at each --horizons value
#   --mode simple   simple goals from a training demo: "descended" (end of the descend stage) from the hover and along
#                   the descent, then "grasped" (end of the grasp stage) from the bottom of the descent
# Prints the planner's first action (and for H > 1 the whole plan) next to the expert's next step.
#
#   PYTHONPATH=<vjepa2> python app/vjepa_hanoi/robot_results/2026-10-13/tools/probe_planner.py --mode horizon \
#       --bundle /scratch/cw5167/checkpoints/vjepa2_ac_hanoi/robot_bundle_firstmove --samples 400 --horizons 1,2,3
# On a 16 GB GPU add --predictor_chunk 100 and run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; at 800
# samples H >= 2 does not fit there (it does on an H200).
import argparse
import os
import time

import h5py
import numpy as np

from app.vjepa_hanoi.goal_planner import GoalPlanner

HERE = os.path.dirname(os.path.abspath(__file__))


def closedness(jaw):
    return float(np.clip((0.0340 - jaw) / 0.0031, 0, 1))


def stage_end(h, episode, stage):
    """Last raw row of motion `stage` in the episode's first game move."""
    o, n = int(h["ep_offset"][episode]), int(h["ep_len"][episode])
    st, mv = h["motion_stage"][o:o + n], h["move_idx"][o:o + n]
    first = mv[np.isin(st, np.arange(1, 10))].min()
    return o + int(np.nonzero((st == stage) & (mv == first))[0].max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("horizon", "simple"), required=True)
    p.add_argument("--bundle", default="/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/robot_bundle_firstmove")
    p.add_argument("--recording", help="recording of the transition: episode 0 = first training demo, 9 = held-out",
                   default="/scratch/cw5167/datasets/hanoi_wm_roundtrip_20260925_171442_AAAA_to_CCCC.h5")
    p.add_argument("--live", default=os.path.join(HERE, "..", "hardware", "one_move_2026-10-13_hover",
                                                  "AAAA-BAAA_hold_r0.npz"), help="hardware run log (step 1 frame)")
    p.add_argument("--transition", default="AAAA>BAAA")
    p.add_argument("--horizons", default="1,2,3")
    p.add_argument("--samples", type=int, default=400, help="CEM samples (paper: 800)")
    p.add_argument("--predictor_chunk", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    gp = GoalPlanner(os.path.join(args.bundle, "config.yaml"), os.path.join(args.bundle, "weights.pt"),
                     protocol="paper", goal_bundle=os.path.join(args.bundle, "goals.npz"),
                     predictor_chunk=args.predictor_chunk)  # fmt: skip
    captured, cem = [], gp.mpc.cem
    gp.mpc.cem = lambda *a, **k: (captured.append(r := cem(*a, **k)), r)[1]  # keep the whole plan
    h = h5py.File(args.recording, "r")
    hov, d_end = stage_end(h, 9, 2), stage_end(h, 9, 3)
    if args.mode == "horizon":
        gp.set_goal(transition=args.transition)  # goals 1-3 of the bundle; probe goal 1
        rows = [("held-out hover", hov, 0), ("held-out, 4 steps into the descent", hov + 32, 0)]
        gnames = ["ring grasped (bundle goal 1)"]
    else:
        g_desc, g_grasp = stage_end(h, 0, 3), stage_end(h, 0, 4)
        gp.set_goal(images=[h["pixels"][g_desc], h["pixels"][g_grasp]])
        rows = [(f"held-out, {k} steps below hover", hov + 8 * k, 0) for k in (0, 3, 6)]
        rows += [(f"held-out, {k} steps after the descent ends", d_end + 8 * k, 1) for k in (0, 2)]
        gnames = ["descended (demo 0, end of descend)", "grasped (demo 0, end of grasp)"]
        print(f"goal frames: demo 0 rows {g_desc} / {g_grasp}; held-out hover row {hov}, descent end row {d_end}")
    starts = []
    if args.live and os.path.exists(args.live):
        z = np.load(args.live, allow_pickle=False)
        starts.append(("live hover (hardware run, step 1)", z["image"][0], z["pose"][0], float(z["jaw"][0]), "", 0))
    for name, r, gi in rows:
        d = (h["state"][r + 8, :3] - h["state"][r, :3]) * 1e3
        dg = closedness(h["proprio"][r + 8, 6]) - closedness(h["proprio"][r, 6])
        starts.append((name, h["pixels"][r], h["state"][r].astype(float), float(h["proprio"][r, 6]),
                       f"expert next: dxyz [{d[0]:+.1f} {d[1]:+.1f} {d[2]:+.1f}] mm, grip {dg:+.2f}", gi))
    for name, image, pose, jaw, expert, gi in starts:
        print(f"\n=== {name}: z {pose[2] * 1e3:.0f} mm, goal '{gnames[gi]}'  {expert}", flush=True)
        for H in [int(x) for x in args.horizons.split(",")]:
            gp.cem_kw.update(rollout=H, samples=args.samples)
            t = time.time()
            out = gp.plan(image, pose, jaw, goal_index=gi, seed=args.seed)
            a = np.array(out["action"])
            print(f"    H={H} ({args.samples}): dxyz [{a[0] * 1e3:+.1f} {a[1] * 1e3:+.1f} {a[2] * 1e3:+.1f}] mm, grip "
                  f"{a[6]:+.2f} -> {out['gripper_intent']:5s} | energy now {out['goal_energy']:.3f}, after plan "
                  f"{out['plan_energy']:.3f} ({time.time() - t:.0f} s)", flush=True)
            if H > 1:
                seq = captured[-1][0].float().cpu().numpy()
                print("          whole plan: " + " | ".join(f"dz {s[2] * 1e3:+.0f} grip {s[6]:+.2f}" for s in seq))


if __name__ == "__main__":
    main()

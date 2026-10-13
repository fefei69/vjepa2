# Why does the planner close the gripper at the hover? Single-frame diagnostic on the cluster (no robot).
#
# At the hover frame of each case's held-out first move (the last row of the approach, stage 2), and at the live hover
# frame of the 2026-10-13 hardware run for AAAA>BAAA, plan toward two goals of the first training demo of the same move:
#   descended   end of stage 3 (jaws open around the ring)
#   grasped     end of stage 4 (= the paper protocol's goal 1)
# with, per horizon:
#   forced open    gripper action held at 0 (cem axis {3: 0}): the best plan that keeps the jaws open
#   forced closed  gripper action held at 1 (cem close_gripper=0): the best plan that closes at once
#   free           the plain planner at initial gripper std 1.0 (repo), 0.3, 0.1, seeds 0 and 1
# and reports each plan's imagined energy to the goal (L1 of the final predicted frame, as CEM ranks samples). If the
# forced-closed plan reaches a lower energy than the forced-open one, the model itself prefers "close" there and no CEM
# sampling setting can remove the hover close; if not, the close is a sampling artefact.
#
#   PYTHONPATH=. python app/vjepa_hanoi/robot_results/2026-10-13/cluster_offline/tools/hover_probe.py --out <jsonl>
import argparse
import json
import os
import time

import h5py
import numpy as np
import torch

from app.vjepa_hanoi.goal_planner import GoalPlanner
from app.vjepa_hanoi.hanoi import hanoi_states
from app.vjepa_hanoi.paper_protocol_eval import goal_rows, moves
from app.vjepa_hanoi.plan_eval import rollout

HERE = os.path.dirname(os.path.abspath(__file__))
K = "/scratch/cw5167/checkpoints/vjepa2_ac_hanoi"
LIVE = os.path.join(HERE, "..", "..", "hardware", "one_move_2026-10-13_hover", "AAAA-BAAA_hold_r0.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fname", default="configs/train/vitg16/hanoi-firstmove-ac-ft-256px-8f.yaml")
    p.add_argument("--checkpoint", default=f"{K}/A_firstmove_vitg_finetune_v1/best.pt")
    p.add_argument("--goal_archive", default=f"{K}/data/expert6_first_move/train.npz")
    p.add_argument("--archive", default=f"{K}/data/expert6_first_move/test.npz")
    p.add_argument("--horizons", default="1,2,3")
    p.add_argument("--samples", type=int, default=800)
    p.add_argument("--gripper_stds", default="1.0,0.3,0.1")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--cases", default="", help="comma-separated transitions (default: all six)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    gp = GoalPlanner(args.fname, args.checkpoint, goal_archive=args.goal_archive, protocol="paper")
    train, train_paths = gp.training_moves()
    test, test_paths, _ = moves(args.archive)
    cases = [c for c in args.cases.split(",") if c] or list(dict.fromkeys(m["transition"] for m in test))
    starts = []  # (case, name, image, pose (6,), jaw)
    for m in test:
        if m["transition"] not in cases:
            continue
        r = int(m["rows"][m["stages"] == 2].max())
        with h5py.File(test_paths[m["file"]], "r") as h:
            starts.append((m["transition"], f"held-out hover (file {m['file']} row {r})", h["pixels"][r],
                           h["state"][r].astype(np.float64), float(h["proprio"][r, 6])))  # fmt: skip
    if "AAAA>BAAA" in cases and os.path.exists(LIVE):
        z = np.load(LIVE)
        starts.insert(0, ("AAAA>BAAA", "live hover (hardware 2026-10-13, step 1)", z["image"][0],
                          z["pose"][0].astype(np.float64), float(z["jaw"][0])))  # fmt: skip

    goals = {}
    for c in cases:
        m = next(m for m in train if m["transition"] == c)
        g = goal_rows(m, 8, (3, 4))
        with h5py.File(train_paths[m["file"]], "r") as h:
            goals[c] = {"descended": gp.encode(h["pixels"][g[1]][None]), "grasped": gp.encode(h["pixels"][g[2]][None])}

    base = dict(gp.cem_kw) | {"samples": args.samples}
    out = open(args.out, "w")
    for case, name, image, pose, jaw in starts:
        s7 = hanoi_states(pose[None], np.full((1, 8), jaw))
        s = torch.as_tensor(s7[None], dtype=torch.float32, device=gp.device)
        z0 = gp.encode(np.asarray(image)[None])
        for gname, goal in goals[case].items():
            for H in [int(x) for x in args.horizons.split(",")]:
                runs = [("forced_open", 1.0, 0, {"axis": {3: 0.0}}, None), ("forced_closed", 1.0, 0, {}, 0)]
                runs += [("free", float(sd), int(seed), {}, None) for sd in args.gripper_stds.split(",")
                         for seed in args.seeds.split(",")]  # fmt: skip
                for mode, gstd, seed, extra, close in runs:
                    kw = base | {"rollout": H, "gripper_std": gstd} | extra
                    torch.manual_seed(seed)
                    t = time.time()
                    with torch.no_grad(), gp._autocast():
                        plan = gp.mpc.cem(context_frame=z0, context_pose=s, goal_frame=goal, world_model=gp.step_fn,
                                          close_gripper=close, **kw)[0].float()  # fmt: skip
                        final = rollout(gp.step_fn, z0, s, plan[None]).float()
                    a = plan.cpu().numpy()
                    rec = {"case": case, "start": name, "z_mm": pose[2] * 1e3, "goal": gname, "H": H, "mode": mode,
                           "gripper_std": gstd, "seed": seed, "samples": args.samples,
                           "first_dxyz_mm": (a[0, :3] * 1e3).round(1).tolist(), "first_grip": float(a[0, 6]),
                           "intent": "close" if np.clip(s7[0, 6] + a[0, 6], 0, 1) > 0.5 else "open",
                           "plan_grip": a[:, 6].round(2).tolist(), "plan_dz_mm": (a[:, 2] * 1e3).round(1).tolist(),
                           "energy_now": float(torch.mean(torch.abs(z0 - goal))),
                           "plan_energy": float(torch.mean(torch.abs(final - goal))), "seconds": time.time() - t}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(f"{case} {name[:12]} {gname:9s} H={H} {mode:13s} std {gstd:.1f} s{seed}: grip "
                          f"{rec['first_grip']:+.2f} {rec['intent']:5s} dxyz {rec['first_dxyz_mm']} energy "
                          f"{rec['plan_energy']:.4f} ({rec['seconds']:.0f} s)", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()

# Offline search for a clean descend + grasp (cluster H200, no robot), 2026-10-14

These experiments follow section 5 of `../RESULTS.md`, in its order. They use the same weights as the robot bundle:
`A_firstmove_vitg_finetune_v1/best.pt`, epoch 5 (`robot_bundle_firstmove/weights.pt`).

The planner is always `mpc_utils.cem` with 800 samples, top-k 10, 10 CEM iterations and maxnorm 0.05. All rows are
open-loop: each row is planned from the recorded frame and state of a held-out first move, and the planned step is
compared with the expert's next step. One model step is 8 raw rows (0.267 s).

**Labels.** Every deviation from the paper protocol is opt-in, and all defaults are unchanged.
- **B**: paper baseline. Goals `paper3` (ring grasped, above the target, final), horizon 1, 800 samples, gripper std 1.0.
- **P**: planner deviation. Horizon > 1 (`--rollout`), `--gripper_std`, or per-stage goals (`--goals stages` /
  server `--subgoals stages`: the end of descend, grasp, lift, transit, insert and release in the same training demo,
  with the median expert steps per stage as the schedule).
- **A**: task aid. `--hold_open` holds the gripper action at 0 while the descend goal is active. `--hold_xy` holds
  dx and dy at 0 while the descend or grasp goal is active. `--hold_grasp` forces closed during lift, transit and
  insert.

## Summary

| | Baseline (B) | Best without aids (P): stages, H=3 | Best overall (A): stages, H=3, hold_xy (+ hold_open, hold_grasp) |
|---|---|---|---|
| gripper at the hover (6 cases) | closes 5/6 | open 6/6 | open 5/6 with hold_xy only; 6/6 with hold_open (by construction) |
| sideways near the rods, median / p90 | 11.1 / 19.9 mm | 6.5 / 11.0 mm | 0 / 0 mm (by construction) |
| descent per step, plan vs expert | −50 vs −13.6 mm | −41 vs −13.6 mm | −45 vs −13.6 mm |
| closes at the bottom (grasp stage) | 0.93 | 0.80–0.83 | 0.93 (hold_xy) / 0.90 (+ hold_open, descend rows) |
| ring kept closed while carrying (lift/transit/insert rows) | — | 108/143 rows; every case opens at least 4 times | 143/143 with hold_grasp (by construction) |

- **The hover close is a planner effect.** The model is close to indifferent to the gripper at H=1–2: forcing the
  gripper closed or open changes the imagined goal energy by ≤0.3% (median). With a near goal (the end of descend)
  and H=3, the model prefers open, so per-stage goals at H=3 fix the close with no aid. The initial gripper std does
  not change this. At H=1, std 0.1 only hides the near-tie (2/6 still close).
- **Sideways drift is not a sampling effect.** With no aid it stays at about 6.5 mm median near the rods, which
  would trip the arm's 3 mm low-lateral guard. Only `hold_xy` (A) removes it.
- **Step size.** The planned descent sits at or near the 50 mm clip in every setting. On the arm, the execution cap
  `--max_speed 0.0625` (16.7 mm per step, `../RESULTS.md` §5.5) has to set the step size. This is an execution limit,
  not a planner change.
- **Clean descend + grasp offline** therefore needs P (stages, H=3) plus A (hold_xy); hold_open is optional at H=3.
  **Carrying the ring** needs A (hold_grasp): without it the planner plans to open on 35 of 143 carry rows, in every
  case.
- **Lift and insert also drift sideways** with no aid for those stages: lift is 13.5 mm against the expert's 0.4 mm,
  and insert is 6.1 mm against 0.1 mm (median |dxy|). Insertion may therefore also trip the low-lateral guard.

## 1. Reproduction (`probes/horizon_probe_800.txt`, `probes/simple_probe_800.txt`)

This uses `../tools/probe_planner.py` at 800 samples with H = 1, 2 and 3. 800 samples fit on one H200 at every
horizon without `--predictor_chunk`. Time per plan: H=1 7 s, H=2 21 s, H=3 41 s.

- At the live hover frame from the hardware run (step 1, z 191 mm), H=1 plans dxyz (+9.8, −31.9, −50.0) mm and grip
  +0.75, i.e. close. This is exactly what the arm did.
- Toward the bundle's goal 1 (ring grasped), H=2 still closes, and H=3 keeps the first action open, at the live and
  the held-out hover alike.
- Toward the end of descend, H=2 is borderline (+0.48 open on the held-out hover, +0.72 close on the live frame), and
  H=3 is open on both.

## 2. Why it closes at the hover (`probes/hover_probe.txt`, `.jsonl`, tool `tools/hover_probe.py`)

The probe uses seven start frames: the six held-out hovers and the live frame. From each one it plans with the
gripper forced open (cem `axis {3: 0}`), forced closed (`close_gripper=0`), and free, with std 1.0 / 0.3 / 0.1 and
seeds 0 and 1.

| Goal, H | forced-closed plan has lower energy | median (closed − open) energy | free planner closes, std 1.0 / 0.3 / 0.1 (of 14) |
|---|---|---|---|
| descended, H=1 | 5/7 | −0.3% | 8 / 8 / 1 |
| descended, H=2 | 4/7 | 0.0% | 9 / 8 / 3 |
| descended, H=3 | 0/7 | +2.2% | 0 / 0 / 0 |
| grasped, H=1 | 7/7 | −0.3% | 13 / 10 / 3 |
| grasped, H=2 | 5/7 | −0.1% | 11 / 10 / 5 |
| grasped, H=3 | 2/7 | +0.2% | 5 / 3 / 1 |

## 3. Descend/grasp sweep (`ppe/descent_*.json`, report `tools/descent_report.py`)

The sweep covers the 78 rows in stages 3 (descend) and 4 (grasp) of the six held-out first moves.

- hover close: the first descend row, i.e. the gripper decision at the hover.
- early close: share of descend rows more than 15 mm above the grasp height that plan "close".
- close at bottom: share of grasp-stage rows that plan "close".
- sideways: planned |dx, dy| within 40 mm of the grasp height while a descend or grasp goal is active. The expert's
  is about 0.
- In every setting, one row per case falls after the time schedule has switched to the next goal while the demo is
  still grasping. Those rows are excluded here: they plan about 70 mm sideways (about 18–20 mm with stages at H=3).

| Setting | Label | Flags | hover close | early close | close at bottom | sideways median / p90 (mm) | descent/step plan vs expert (mm) |
|---|---|---|---|---|---|---|---|
| base_h1 | **B** | (none) | 5/6 | 0.69 | 0.93 | 11.1 / 19.9 | −50 vs −13.6 |
| base_h2 | P | `--rollout 2` | 4/6 | 0.42 | 0.83 | 10.6 / 15.7 | −50 |
| base_h3 | P | `--rollout 3` | 4/6 | 0.42 | 0.80 | 6.6 / 9.8 | −31.2 |
| g03_h1 | P | `--gripper_std 0.3` | 4/6 | 0.64 | 0.93 | 11.1 / 20.1 | −50 |
| g03_h2 | P | `--gripper_std 0.3 --rollout 2` | 4/6 | 0.42 | 0.90 | 10.4 / 15.8 | −50 |
| g01_h1 | P | `--gripper_std 0.1` | 2/6 | 0.19 | 0.70 | 12.4 / 21.0 | −50 |
| stages_h1 | P | `--goals stages` | 4/6 | 0.28 | 1.00 | 10.5 / 18.1 | −50 |
| stages_h2 | P | `--goals stages --rollout 2` | 5/6 | 0.25 | 0.93 | 8.5 / 15.7 | −50 |
| **stages_h3** | P | `--goals stages --rollout 3` | **0/6** | **0.00** | 0.83 | 6.5 / 11.0 | −41.1 |
| stages_g03_h1 | P | `--goals stages --gripper_std 0.3` | 4/6 | 0.25 | 1.00 | 10.2 / 15.6 | −50 |
| stages_g03_h2 | P | `... --rollout 2` | 5/6 | 0.22 | 0.97 | 8.2 / 15.5 | −50 |
| stages_g03_h3 | P | `... --rollout 3` | 0/6 | 0.00 | 0.87 | 6.4 / 10.5 | −41.6 |
| aid_xy_h1 | A | `--hold_xy` (paper3 goals) | 6/6 | 0.86 | 0.90 | 0 / 0 | −50 |
| aid_stages_open_h1 | A | `--goals stages --hold_open` | 0/6 | 0.00 | 1.00 | 8.2 / 16.1 | −50 |
| aid_stages_xy_h1 | A | `--goals stages --hold_xy` | 6/6 | 0.61 | 0.97 | 0 / 0 | −50 |
| **aid_stages_open_xy_h1** | A | `--goals stages --hold_open --hold_xy` | **0/6** | **0.00** | **0.97** | **0 / 0** | −50 |
| aid_stages_open_xy_h2 | A | `... --rollout 2` | 0/6 | 0.00 | 0.97 | 0 / 0 | −50 |
| **aid_stages_open_xy_h3** | A | `--goals stages --hold_open --hold_xy --rollout 3` | **0/6** | **0.00** | 0.90 | **0 / 0** | −41.6 |

At the first descend row, stages_h3 gives: AAAA>BAAA open (−0.26), AAAA>CAAA open (+0.00), BBBB>ABBB open (−0.69),
BBBB>CBBB open (−0.63), CCCC>ACCC open (−0.34), CCCC>BCCC open (−0.53).

## 4. Whole move, all six cases (`ppe/all_*.json`, report `tools/move_report.py`)

Every row of the six held-out first moves is planned: 324 rows, stages 1–8.
- grip ok: the planned gripper intent matches the expert's.
- lat err / dz err: median |planned − expert| step, in mm.
- cos: 3D cosine on rows where the expert moves at least 5 mm.

In the paper-protocol robot runs, stages 1–2 are replaced by the runner's hover start. Their rows here use the
descend goal from the move start, so treat them as context only.

| Stage | rows | stages_h3 (P): grip ok / lat err / dz err / cos | aid_stages_xy_h3 (A: hold_xy): grip ok / lat err / dz err / cos |
|---|---|---|---|
| 3 descend | 48 | 0.96 / 8.5 / 16.2 / 0.96 | 0.92 / 0.2 / 21.7 / 1.00 |
| 4 grasp | 30 | 0.60 / 7.6 / 0.8 / – | 0.67 / 0.2 / 1.2 / – |
| 5 lift | 38 | 0.71 / 13.3 / 18.1 / 0.94 | same (hold_xy is off from stage 5) |
| 6 transit | 63 | 0.87 / 38.8 / 4.8 / 0.70 | same |
| 7 insert | 42 | 0.62 / 6.2 / 8.4 / 0.94 | same |
| 8 release | 15 | 0.40 / 5.4 / 1.8 / – | same |

- Carrying (stages 5–7, where the expert is always closed): the plan intends "open" on 11/38 lift rows, 8/63 transit
  rows and 16/42 insert rows. Every case has 4–9 such rows, so on the arm the ring would be dropped. `--hold_grasp`
  (A) is required.
- Grasp-stage grip ok is 0.60–0.67 because the plan's close comes earlier or later than the expert's within the
  5-step grasp stage. close at bottom (above) is 0.80–0.93.
- In whole-move mode, the descend run's hover decision (descent_report `--prefix all_`) was open 6/6 for stages_h3
  and 5/6 for aid_stages_xy_h3 (CCCC>ACCC +0.54 close). With x/y held, the gripper is again a near-tie at that one
  hover. hold_open removes it by construction (section 3).

## 5. Recommended setups for the arm (the user runs it; nothing here touched the robot)

The bundle's `goals.npz` was re-exported with stage frames:

```
python -m app.vjepa_hanoi.export_robot_bundle --goals_only --fname configs/train/vitg16/hanoi-firstmove-ac-ft-256px-8f.yaml \
  --checkpoint $K/A_firstmove_vitg_finetune_v1/best.pt --goal_archive $K/data/expert6_first_move/train.npz \
  --out $K/robot_bundle_firstmove
```

- The new goals.npz has sha256 `cc13371b…62f7f3`, 4.4 MB. Copy it and MANIFEST.json to the robot; weights.pt is
  unchanged.
- The paper3 keys are byte-identical to the 2026-10-08 file, which is backed up in `robot_bundle_firstmove_goals_2026-10-08/`.
- The new keys are stage_frames, stage_frame_sources, stage_goal_stages and stage_schedules. Stage schedules:
  [24,5,8,8,8,3] for AAAA>BAAA and CCCC>BCCC, [24,5,8,16,7,4] for AAAA>CAAA and CCCC>ACCC, and [21,5,8,8,8,4] for the
  BBBB cases.
- From the hover, the runner subtracts the approach (16 steps for pegs A and C, 13 for B), which leaves 8 descend steps.

New runner variants in `one_move_test.py` (`--variants`):

| variant | label | server options |
|---|---|---|
| faithful | B | paper3 (start the server with the default --rollout 1) |
| stages | P | subgoals stages (start the server with `--rollout 3`) |
| stages_xy | P + A | stages, hold_xy, hold_grasp |
| stages_open_xy | P + A | stages, hold_open, hold_xy, hold_grasp |

Suggested arm order:
1. `faithful` (the baseline, already run on 2026-10-13).
2. `stages` with the server at `--rollout 3`: the best non-aided setting, expected to fail on sideways and carrying.
3. `stages_xy` or `stages_open_xy` at `--rollout 3`: expected to descend and grasp cleanly.

Run all of them with `--max_speed 0.0625`.

**Speed caveat.** H=3 at 800 samples takes about 41 s per step on an H200. On the robot's RTX 5080 (16 GB), the
server notes say H ≥ 2 needs `--predictor_chunk 100`. Chunking gives bit-identical plans and only lowers memory.
With chunking, H=3 at 400 samples took about 80 s per step, so 800 samples should take roughly 160 s. That is above
the client's default 120 s timeout, so raise it, e.g. `PLANNER_TIMEOUT_S=300` in the runner's environment. 400 samples avoids that, but it is a further deviation:
everything here was measured at 800 samples. A move from the hover is about 36 steps (8+5+8+8+8+3), i.e. 50–95 min
of planning on the 5080.

## Commands

```
T=app/vjepa_hanoi/robot_results/2026-10-13/cluster_offline/tools
sbatch --export=ALL,TASK=probes $T/offline.sbatch                          # section 1-2
sbatch --export=ALL,TASK=ppe,SETTINGS="base_h1 stages_h3 ..." $T/offline.sbatch          # section 3 (rows in stages 3-4)
sbatch --export=ALL,TASK=ppe,ROWS=all,SETTINGS="stages_h3" $T/offline.sbatch             # section 4 (whole move)
PYTHONPATH=. python $T/descent_report.py app/vjepa_hanoi/robot_results/2026-10-13/cluster_offline/ppe
PYTHONPATH=. python $T/move_report.py    app/vjepa_hanoi/robot_results/2026-10-13/cluster_offline/ppe
```

The flags for each named setting are in `offline.sbatch::setting()`. paper_protocol_eval's flags: `--goals
paper3|final_only|stages`, `--rollout`, `--samples`, `--gripper_std`, `--gripper_clip`, `--hold_open`, `--hold_xy`,
`--hold_grasp`, `--eval_stages`, `--predictor_chunk`.

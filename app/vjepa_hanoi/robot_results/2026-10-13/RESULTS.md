# V-JEPA 2-AC one-move test on the real arm: results and the open problem (2026-10-13)

Case tested: **AAAA>BAAA** (ring 1 from peg A to peg B), variant `hold`, bundle `robot_bundle_firstmove` (weights
sha256 `b60ad194…d320`, epoch 5). Robot machine: Trossen WXAI, RealSense on thor2, RTX 5080 (16 GB) running the
planning server locally. Code: branch `hanoi-ac`, commits d87c15a (server) and 6271bb6 (runner); the hardware runs
below ran that code before it was committed (their session files show HEAD 2b4a2f0 plus the changed files).

**Bottom line.** The published protocol (horizon 1, three goal images, 800 CEM samples) fails at the first stage on the
arm: at step 1, above the source peg, the planner **closes the gripper** (gripper action +0.75, the sampling limit),
about 105 mm above the ring, then descends with closed jaws, with large random sideways components, until the 3 mm
low-lateral guard stops it. No planner parameter tried offline (maxnorm, step cap, horizon 1-3, simpler subgoals)
removes both failure modes: **(1) a premature close at the hover, (2) sideways drift near the rods.**

## 1. Hardware runs

| Run (`hardware/`) | Start | Server step cap | Steps | Outcome |
|---|---|---|---|---|
| `one_move_2026-10-11` | episode start (414, 16, 191) mm | 26.7 mm | 0 | stopped by the operator (Ctrl-C) before step 1 |
| `one_move_2026-10-13` | episode start | 8 mm (`--max_speed 0.03`) | 2 | operator Ctrl-C; closed at step 1 in mid-air |
| `one_move_2026-10-13_hover` | hover above peg A (498.4, -57.2, 191.1) mm | 8 mm | 4 | `UnsafeStep`: 5.8 mm sideways below the low floor |

All: paper planner (rollout 1, 800 samples, 10 CEM steps, top 10, maxnorm 0.05, momentum 0.15), `--predictor_chunk
200`, seed 37195. Hover run step by step (plan = CEM's first action before execution safety; exec = executed delta):

| Step | Tool z | Planned dxyz (mm) | Gripper | Executed (mm) | Goal-1 energy |
|---|---|---|---|---|---|
| 1 | 191 | (+9.7, -31.6, -50.0) | +0.75 -> close (jaw then reads 0.0037: closed on air) | (+1.1, -0.8, -7.9) | 0.531 |
| 2 | 183 | (-6.9, -40.5, -50.0) | +0.68 close | (-1.1, -0.7, -7.9) | 0.532 |
| 3 | 175 | (-14.0, -18.2, -50.0) | +0.61 close | (-2.2, -0.5, -7.7) | 0.528 |
| 4 | 167 | (-17.8, +50.0, -50.0) | +0.74 close | (-2.0, +5.5, -5.5) -> guard stop | 0.533 |

- z is pinned at -50 mm (the sampling limit) every step; y jumps between -40 and +50 mm; the energy stays flat.
- Episode-start run: steps 1-2 planned (+50.0, -49.9, -48.5) and (+50.0, -50.0, -34.9) mm, gripper +0.74/+0.75.
- **The input pipeline is not the problem.** The live hover frame and the recorded hover frame of the held-out demo
  differ by 5.8 grey levels mean-abs (10.2 with RGB swapped), same crop: `figures/live_vs_recorded.jpg` (live step 1,
  recorded hover, goal 1, live step 4). Offline, planning from the live step-1 frame reproduces the robot's step 1
  exactly (+9.8, -32.2, -50.0) mm, +0.75 (`offline/horizon_probe.txt`).
- Each `.npz` has every step (pose, jaw, action, executed delta, energies, intent, goal index, plan seconds, move
  time and class, grasp reading) with its 224 px camera frame; `.steps.jsonl` is the same per step without images.

## 2. How the planner works here (notebooks/utils/mpc_utils.py::cem via goal_planner.py --protocol paper)

- Horizon 1 model step = 0.267 s (data: every 8th 30 Hz row, 3.75 Hz). Receding horizon: replan every step.
- Actions: [dx, dy, dz] ~ N(0, 50 mm) per axis clipped to +-maxnorm (50 mm); rotation fixed 0; gripper (change in
  closedness, 0 open .. 1 closed) ~ **N(0, 1.0) clipped to +-0.75**. Final action = CEM mean after 10 iterations.
- Server: intent = close if closedness + a_grip > 0.5; xyz clamped to the training-data box, then capped at
  max_speed x 0.267 s; orientation fixed. Planning takes about 25 s per step on the RTX 5080 (about 7 s on an H200).
- Goals (paper protocol): 3 frames of the first training demo of the same transition: end of grasp, end of transit,
  end of release; switched by step count. From the hover the expert schedule is [13, 16, 11] (demos reach the hover
  15.9 steps after their first row for source pegs A and C, 13.2 for B; the hover is tower_hanoi's
  `grasp_position(peg)` at transit height, matching the demos within 0.4 mm). `figures/goal_images.jpg` shows the
  start (hover) and the three goals for all six cases.
- Expert stage durations from the hover, AAAA>BAAA (model steps): descend 7.9 (105 mm, median 16.5 mm/step, max
  ~24) | grasp 5.2 | lift 7.9 | transit 7.9 (15.8 for A<->C) | insert 7.9 | release 3.8.

## 3. Offline experiments (local RTX 5080, same bundle)

**maxnorm sweep** (`offline/maxnorm_sweep/`, `paper_protocol_eval.py`, 72 held-out rows of the six first moves,
same rows and seeds per setting; held-out archives rebuilt locally from the recordings, matching the bundle's goal
frames and schedules exactly). Expert step: median 7.3 mm, p90 17.5, max 24.8.

| maxnorm | y-z direction right | 3D cos > 0 | planned step (median) | planned when the expert is still | executed error at the 26.7 mm cap |
|---|---|---|---|---|---|
| 50 mm | 0.93 | 0.82 | 65.8 mm (5.8x expert) | 47.3 mm | 22.7 mm |
| 35 mm | 0.97 | 0.84 | 51.6 mm | 43.6 mm | 22.6 mm |
| 25 mm | 0.95 | 0.78 | 40.1 mm | 32.0 mm | 23.1 mm |
| 15 mm | 0.93 | 0.80 | 22.8 mm (2.0x) | 20.8 mm | 21.1 mm |

Direction is roughly right; magnitude always pushes to the sampling limit. Capping the executed step (from the saved
rows): cap 17.5 mm -> ~16 mm error, 10 -> 10, 7.5 -> 7.6; planned direction with the expert's exact size -> 6.0-7.5;
not moving at all -> 7.3 mm. Gripper sign accuracy 1.00 but on only 4 gripper events; 49-79% of rows have the
gripper action at +-0.75.

**Horizon probe, goal 1 "ring grasped"** (`offline/horizon_probe.txt`, 400 samples, seed 7; at 800 samples H >= 2
does not fit in 16 GB). Times: H=1 12 s, H=2 39 s, H=3 80 s per plan.

| Start | Expert next | H=1 | H=2 | H=3 |
|---|---|---|---|---|
| live hover (z 191) | - | close +0.75, y -32 | close +0.74, sideways ~5 mm | close +0.50 (borderline), sideways ~21 mm |
| recorded hover (z 191) | still, open | close +0.75 | close +0.51 | open +0.49; plan open/open/open |
| 4 steps into the descent (z 138) | down 24 mm, open | close +0.74 | open -0.27, straight down | open -0.72; plan: down, down, then close |

**Simple goals** (`offline/simple_goals_probe.txt`, goals from demo 0: "descended" = end of descend, jaws open around
ring 1 at 87 mm; "grasped" = end of grasp; H=1 at 800 and H=2 at 400 samples).

| Start | Goal | Expert next | Planner |
|---|---|---|---|
| hover, live and recorded (z 191) | descended | still | down 50 mm, but **close** (+0.75 / +0.63) |
| 3 steps down (z 165) | descended | down 24, open | down, open, but **20-50 mm sideways** |
| 6 steps down (z 98) | descended | down 10, open | down 13-16 mm, open, small sideways (slows near the goal) |
| end of descent (z 87) | grasped | close | **close**, no z move, but **11-15 mm sideways** |

## 4. Diagnosis

- The model does encode "down" and "close at the bottom": with a near goal it moves down and slows, and from the
  bottom it closes.
- **Premature close at the hover**: present for every goal tried (even "descended", whose jaws are open), at H=1 and
  H=2, borderline at H=3. Likely a learned correlation: in the data a close happens only at grasp height, so a close
  action makes the predicted frame look "lower", i.e. closer to any low goal. Contributing: the gripper's initial
  std 1.0 clipped at +-0.75 puts ~45% of first-iteration samples at the limit, and +0.5 already means close.
- **Sideways drift**: the energy barely penalizes x/y; the expert never moves sideways during descend/grasp.
- **Distance**: goal 1 is 13 expert steps from the hover; horizon 1-3 cannot see it, and the model was trained on
  2-step rollouts (`loss.auto_steps: 2`).

## 5. Suggested next experiments (cluster, offline first)

1. Reproduce: `probe_planner.py --mode horizon` and `--mode simple` (section 6), then `paper_protocol_eval.py` on all
   six cases with `--rollout 2/3 --samples 400/800` (the H200 fits 800 at H >= 2).
2. Gripper sampling, a plain CEM parameter rather than a task aid: smaller initial gripper std (e.g. 0.3) or a wider
   clip; check whether the hover close disappears with the "descended" goal.
3. Per-stage subgoals (end of descend, grasp, lift, transit, insert, release; 4-8 steps apart) with H=2-3.
4. If 1-3 do not fix it (these are task aids, outside the paper baseline): hold the gripper open while "descended" is
   active (mirror of `--hold_grasp`), and hold x/y during descend/grasp via the planner's `axis` option.
5. On the arm, the descent needs about 16.5 mm per step to keep the 13-step goal-1 budget: use
   `--max_speed 0.0625` (16.7 mm cap), not 0.03 (8 mm, which needs 13 steps for the descent alone).

## 6. Files and commands

- `hardware/`: the three runs (session file with server /health and code state, results.jsonl, per-step npz/jsonl).
- `offline/`: maxnorm sweep JSON + per-row npz + `summary.txt`; horizon and simple-goal probe outputs.
- `figures/`: `goal_images.jpg` (start + 3 goals, 6 cases), `live_vs_recorded.jpg`.
- `tools/probe_planner.py` (both probes; cluster paths by default):
  `PYTHONPATH=. python app/vjepa_hanoi/robot_results/2026-10-13/tools/probe_planner.py --mode simple --samples 800 --horizons 1,2,3`
- `tools/sweep_analyze.py <dir with ppe_maxnorm_*.json>`: the sweep tables above (including the step-cap table).
- `tools/run_ppe.py`: runs `paper_protocol_eval.py` with predictor chunking (16 GB GPUs only).
- `tools/build_first_move_archives.py`: rebuilds expert6_first_move train/test from the six recordings when the
  cluster archives are not available (row counts checked against the Cosmos metadata).

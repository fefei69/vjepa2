# V-JEPA 2-AC one-move test on the real arm: results and the open problem (2026-10-13)

Case tested: **AAAA>BAAA** (ring 1 from peg A to peg B), variant `hold` (the runner sends `hold_grasp=true` per run:
gripper held closed while goal 2 is active, so every failure below, all under goal 1, is the same as `faithful`).
Bundle `robot_bundle_firstmove` (weights sha256 `b60ad194…d320`, epoch 5). Robot machine: Trossen WXAI, RealSense on
thor2, RTX 5080 (16 GB) running the planning server locally.

Code: the runs used uncommitted working trees on top of 2b4a2f0 (session diff sha256 `fe54f307…` at 15:54 and
`ffc60852…` at 16:48; the 10-11 run predates the `--execute` gate and records no diff). Their planning code equals
d87c15a except for its memory/override options; the runner/safety code was revised before 6271bb6.

**Bottom line.** The published protocol (horizon 1, three goal images, 800 CEM samples) fails at the first stage on the
arm: at step 1, above the source peg, the planner **closes the gripper** (gripper action +0.75, the sampling limit)
about 105 mm above the ring, then descends with closed jaws while its x/y actions jump between large values, until
the 3 mm low-lateral guard stops it. No planner setting tried offline (maxnorm, step cap, horizon 1-3, simpler
subgoals) removes both failure modes: **(1) a premature close at the hover, (2) large sideways components from the
hover onward** (31-41 mm at H=1 at the hover, 11-57 mm lower down).

## 1. Hardware runs

| Run (`hardware/`) | Start | Server step cap | Steps | Outcome |
|---|---|---|---|---|
| `one_move_2026-10-11` | episode start (414, 16, 191) mm | 26.7 mm | 0 | operator Ctrl-C before step 1 (metadata only) |
| `one_move_2026-10-13` | episode start | 8 mm (`--max_speed 0.03`) | 2 | operator Ctrl-C; closed at step 1 in mid-air |
| `one_move_2026-10-13_hover` | hover above peg A (498.4, -57.2, 191.1) mm | 8 mm | 4 | `UnsafeStep`: 5.8 mm sideways below the low floor |

All: paper planner (rollout 1, 800 samples, 10 CEM steps, top 10, maxnorm 0.05, momentum 0.15), `--predictor_chunk
200`. Run seed 37195; step k plans with CEM seed 37195 x 1000 + k (in `.steps.jsonl`). Hover run, step by step
(plan = CEM's first action before execution safety; exec = executed delta):

| Step | Tool z | Planned dxyz (mm) | Gripper | Executed (mm) | Goal-1 energy |
|---|---|---|---|---|---|
| 1 | 191 | (+9.7, -31.6, -50.0) | +0.75 -> close (jaw then reads 0.0037: closed on air) | (+1.1, -0.8, -7.9) | 0.531 |
| 2 | 183 | (-6.9, -40.5, -50.0) | +0.68 close | (-1.1, -0.7, -7.9) | 0.532 |
| 3 | 175 | (-14.0, -18.2, -50.0) | +0.61 close | (-2.2, -0.5, -7.7) | 0.528 |
| 4 | 167 | (-17.8, +50.0, -50.0) | +0.74 close | (-2.0, +5.5, -5.5) -> guard stop | 0.533 |

- z is at -50 mm (the sampling limit) every step; y swings between -40.5 and +50 mm; the energy stays flat.
- Episode-start run: steps 1-2 planned (+50.0, -49.9, -48.5) and (+50.0, -50.0, -34.9) mm, gripper +0.74/+0.75.
- **The input pipeline is not the problem.** The live hover frame and the recorded hover frame of the held-out demo
  (row 70355) differ by 5.8 grey levels mean-abs (10.2 with RGB swapped), same crop: `figures/live_vs_recorded.jpg`.
  Offline, planning from the live step-1 frame (400 samples, seed 7) gives (+9.7, -31.6, -50.0) mm, +0.75, plan
  energy 0.410; the robot's step 1 (800 samples, seed 37195001) was (+9.7, -31.6, -50.0), +0.75, 0.4095.
- Logs: each `.npz` has every step (pose, jaw, action, executed delta, energies, intent, goal index, plan seconds,
  move time and class, grasp reading) with its 224 px frame. `.steps.jsonl` has the per-step fields without images
  and grasp reading, plus target pose and seed. The 10-11 run has metadata only.

## 2. How the planner works here (notebooks/utils/mpc_utils.py::cem via goal_planner.py --protocol paper)

- Horizon 1 model step = 0.267 s (data: every 8th 30 Hz row, 3.75 Hz). Receding horizon: replan every step.
- Sampling vector [dx, dy, dz, grip] (rotation fixed 0). xyz ~ N(0, maxnorm) per axis, clipped to +-maxnorm: the
  initial std *is* maxnorm, so the maxnorm sweep changes both. grip (change in closedness, 0 open .. 1 closed) ~
  **N(0, 1.0), clipped to +-0.75**. Both are hard-coded in `cem()` (no flag). The returned action is the CEM mean
  after 10 iterations; a gripper mean with |g| < 0.25 is returned as 0 (`round_small_elements`).
- Server: intent = close if clip(closedness + grip, 0, 1) > 0.5; xyz clamped to the training-data box, then capped at
  max_speed x 0.267 s; orientation fixed. Planning takes about 25 s per step on the RTX 5080 (800 samples, H=1).
- Goals (paper protocol): 3 frames of the first training demo of the same transition (end of grasp / transit /
  release), switched by step count. From the hover the expert schedule is [13, 16, 11]: demos reach the hover 15.9
  steps after their first row for source pegs A and C, 13.2 for B; the hover is tower_hanoi's `grasp_position(peg)`
  at transit height, matching the demos within 0.4 mm. `figures/goal_images.jpg`: start + 3 goals, all six cases.
- Expert stage durations from the hover, AAAA>BAAA (model steps): descend 7.9 (105 mm, median 16.5 mm/step, max
  ~24) | grasp 5.2 | lift 7.9 | transit 7.9 (15.8 for A<->C) | insert 7.9 | release 3.8.

## 3. Offline experiments (local RTX 5080, same bundle)

**maxnorm sweep** (`offline/maxnorm_sweep/`, `paper_protocol_eval.py --max_rows 72`: 72 rows spread over the six
held-out first moves, same rows and seeds per setting; held-out/train archives rebuilt locally from the recordings,
reproducing the bundle's goal frames and schedules exactly). Rows step from each move's **first row** (episode start,
schedule [29,16,11]), not from the hover; for hover-start numbers use the rows with stage >= 3 in `*_rows.npz`.
Expert step: median 7.3 mm, p90 17.5, max 24.8.

| maxnorm | y-z direction right | 3D cos > 0 | planned step (median) | median per-row ratio to expert (moving rows) | rows with an axis at the clip | executed error at the 26.7 mm cap |
|---|---|---|---|---|---|---|
| 50 mm | 0.93 | 0.82 | 65.8 mm | 5.8 | 0.67 | 22.7 mm |
| 35 mm | 0.97 | 0.84 | 51.6 mm | 4.7 | 0.74 | 22.6 mm |
| 25 mm | 0.95 | 0.78 | 40.1 mm | 3.4 | 0.79 | 23.1 mm |
| 15 mm | 0.93 | 0.80 | 22.8 mm | 2.0 | 0.82 | 21.1 mm |

Direction is roughly right; the planned magnitude scales with maxnorm (median 1.3-1.6x maxnorm) and is far above the
expert's, even on rows where the expert is still (planned 21-47 mm). Capping the executed step (from the saved rows):
cap 16.7 mm -> 15-16 mm median error, 10 -> 10, 8 -> 8; planned direction at the expert's exact size -> 6.0-7.5; not
moving at all -> 7.3 mm. Gripper sign accuracy 1.00 but on only 4 gripper events; 49-79% of rows have the gripper
action at +-0.75.

**Horizon probe, goal 1 "ring grasped"** (`offline/horizon_probe.txt`, 400 samples, seed 7, held-out rows 70355
and 70387; at 800 samples H >= 2 does not fit in 16 GB). Times on the RTX 5080: H=1 12 s, H=2 39 s, H=3 80 s.

| Start | Expert next | H=1 | H=2 | H=3 |
|---|---|---|---|---|
| live hover (z 191) | - | close +0.75; 33 mm sideways | close +0.74; 5 mm sideways | close +0.50 (borderline); 21 mm sideways |
| recorded hover (z 191) | still, open | close +0.75; 32 mm sideways | close +0.51; 12 mm sideways | open +0.49; 9 mm sideways; plan stays open |
| 4 steps into the descent (z 138) | down 24 mm, open | close +0.74 | open -0.27, straight down | open -0.72 but 26 mm sideways and only 29 mm down; plan closes at its step 2 |

**Simple goals** (`offline/simple_goals_probe.txt`, goals from demo 0: "descended" = end of descend, jaws open around
ring 1 at 87 mm; "grasped" = end of grasp; H=1 at 800 samples, H=2 at 400; held-out rows from the last approach row
70354 on).

| Start | Goal | Expert next | Planner H=1 / H=2 |
|---|---|---|---|
| live hover (z 191) | descended | - | down 50, **close** +0.75 / +0.75; sideways 41 / 1 mm |
| recorded hover (z 191) | descended | still | down 50, **close** +0.75 / +0.63; sideways 31 / 11 mm |
| 3 steps down (z 165) | descended | down 24, open | down 50, open, **57 / 32 mm sideways** |
| 6 steps down (z 98) | descended | down 10, open | down 16 / 13 mm, open, 7 / 4 mm sideways (slows near the goal) |
| end of descent (z 87) | grasped | still (closes 2 steps later) | **close** +0.63 / +0.58, no z move, 12 / 11 mm sideways |
| 2 steps later (z 86) | grasped | close +0.70 | **close** +0.73 / +0.67, 18 / 17 mm sideways |

## 4. Diagnosis

- The model does encode "down" and "close at the bottom": with a near goal it moves down and slows, and from the
  bottom it closes.
- **Premature close at the hover**: present for every goal tried (even "descended", whose jaws are open), at H=1 and
  H=2, borderline at H=3. Likely a learned correlation: in the data a close happens only at grasp height, so a close
  action makes the predicted frame look "lower", i.e. closer to any low goal. Contributing: the gripper's initial
  std 1.0 clipped at +-0.75 puts ~45% of first-iteration samples at the limit, and +0.5 already means close.
- **Sideways components**: the energy barely penalizes x/y; the expert never moves sideways during descend/grasp.
- **Distance**: goal 1 is 13 expert steps from the hover; horizon 1-3 cannot see it, and the model was trained on
  2-step rollouts (`loss.auto_steps: 2`).

## 5. Suggested next experiments (cluster, offline first)

1. Reproduce: `tools/probe_planner.py` (section 6). The probe outputs here came from earlier scratch scripts with the
   rows noted above; the tool's horizon mode starts from the last approach row (70354/70386), so expect small
   differences. Then `paper_protocol_eval.py` with `--rollout 2/3` and `--samples 400/800` (section 6 command).
2. Gripper sampling (a CEM change, not a task aid; needs code): add `gripper_std=1.0, gripper_clip=0.75` keyword
   arguments to `mpc_utils.cem` (defaults unchanged), pass them through `cem_kw`, and test e.g. std 0.3 with
   `probe_planner --mode simple` at the hover rows (goal "descended"). A clip above 1.0 changes nothing: the server
   clips closedness to [0, 1] and closes above 0.5.
3. Per-stage subgoals: stage ends 3 descend, 4 grasp, 5 lift, 6 transit, 7 insert, 8 release (4-8 steps apart), with
   the median expert steps per stage as the schedule and H=2-3. Needs code: `paper_protocol_eval.py` fixes
   `GOAL_STAGES = (4, 6)` + the final stage, and the bundle holds 3 frames per transition.
4. Only if 1-3 fail (task aids, outside the paper baseline): hold the gripper open while "descended" is active
   (`cem_kw["axis"] = {3: 0.0}`, since the sampling vector is [dx, dy, dz, grip]; `close_gripper` can only force a
   close), and hold x/y during descend/grasp (`cem_kw["axis"] = {0: 0.0, 1: 0.0}`; only `--fix_x` exists as a flag).
5. On the arm, the descent needs about 16.5 mm per step to fit the 13-step goal-1 budget: `--max_speed 0.0625`
   (16.7 mm cap), not 0.03 (8 mm, which needs 13 steps for the descent alone).

## 6. Files and commands

- `hardware/`: the three runs (session file with server /health and code state, results.jsonl, per-step npz/jsonl).
- `offline/`: maxnorm sweep JSON + per-row npz + `summary.txt`; horizon and simple-goal probe outputs.
- `figures/`: `goal_images.jpg` (start + 3 goals, 6 cases), `live_vs_recorded.jpg` (live step 1 | recorded hover |
  goal 1 | live step 4).
- `tools/probe_planner.py`: both probes; defaults are the cluster bundle and the AAAA_to_CCCC recording, and `--live`
  defaults to the committed hover-run npz (`--live ''` disables it). `--transition`, `--recording` and `--live` go
  together: the recording must hold that transition (episode 0 = first training demo, 9 = held-out); the live frame
  exists only for AAAA>BAAA.
- `tools/sweep_analyze.py <dir>`: the sweep tables above (labels rows by maxnorm; adapt for rollout/samples sweeps).
- `tools/run_ppe.py`: `paper_protocol_eval.py` with predictor chunking, for 16 GB GPUs only.
- `tools/build_first_move_archives.py`: only for machines without the cluster archives (it rebuilds train/test from
  the recordings using the Cosmos `multitask_v6/metadata.json`; row counts checked against it).

On the cluster (paths as used on the cluster before; the H200 numbers are expectations, not measured here: if memory
runs out, add `--predictor_chunk 400` to the probe or use `tools/run_ppe.py`):

```bash
cd /scratch/cw5167/workspace/vjepa2 && git pull
K=/scratch/cw5167/checkpoints/vjepa2_ac_hanoi; B=$K/robot_bundle_firstmove   # MANIFEST sha256 b60ad194...d320
# probes (one GPU, a few minutes):
PYTHONPATH=. .venv/bin/python app/vjepa_hanoi/robot_results/2026-10-13/tools/probe_planner.py --mode simple \
    --bundle $B --samples 800 --horizons 1,2,3
# row-paired with the sweep here (72 rows):
PYTHONPATH=. .venv/bin/python -m app.vjepa_hanoi.paper_protocol_eval --fname $B/config.yaml --checkpoint $B/weights.pt \
    --archive $K/data/expert6_first_move/test.npz --goal_archive $K/data/expert6_first_move/train.npz \
    --max_rows 72 --rollout 2 --samples 800 --out $K/paper_protocol/firstmove_2026-10-13/ppe_r2_s800.json
```

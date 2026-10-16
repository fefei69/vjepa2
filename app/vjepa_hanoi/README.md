# V-JEPA 2-AC on the Tower-of-Hanoi play_k5 split

Action-conditioned V-JEPA 2 world models trained on exactly the training data of the pi0.5 and Cosmos play runs.
The data contract is in `/scratch/cw5167/workspace/openpi/docs/hanoi_play_k5_dataset_handover.md`. The training loop
is a copy of `app/vjepa_droid/train.py` (frozen encoder; teacher-forcing + 2-step rollout L1 on latents).

| Run | Config | Encoder (frozen) | Predictor |
|---|---|---|---|
| A | `configs/train/vitg16/hanoi-play-k5-ac-ft-256px-8f.yaml` | V-JEPA 2 ViT-g from `vjepa2-ac-vitg.pt` (`target_encoder`) | fine-tuned from the released DROID-trained AC predictor |
| B | `configs/train/vitl16/hanoi-play-k5-ac-scratch-256px-8f.yaml` | V-JEPA 2 ViT-L, `vitl.pt` `encoder` = `facebook/vjepa2-vitl-fpc64-256` (292/292 tensors identical) | from scratch |

A and B differ in both encoder and initialization. No ViT-L AC predictor was released, so a fine-tuned ViT-L run is
not possible.

## Run

```bash
sbatch --export=ALL,CONFIG=configs/train/vitg16/hanoi-play-k5-ac-ft-256px-8f.yaml app/vjepa_hanoi/train.sbatch     # A
sbatch --export=ALL,CONFIG=configs/train/vitl16/hanoi-play-k5-ac-scratch-256px-8f.yaml app/vjepa_hanoi/train.sbatch # B
python -m app.vjepa_hanoi.eval --fname <config> --checkpoint <folder>/best.pt --split test                        # GPU
```

Both runs use 2x H200 (`train.sbatch` default) with the per-GPU batch set so the global batch is A 32 / B 64.
Each run writes `log_r*.csv`, `val_log.csv`, `best.pt` (lowest validation loss), `latest.pt` and `e*.pt` into its
`folder`. Resubmitting the same config resumes from `latest.pt`.

## Data (same as the policy baselines)

- Split: the manifest split via the openpi archives `play_{train,val,test}.npz`. Training uses only the training
  segments (78 walks, 14 one-move crops, 4 expert clips). The 10 validation walks are used only for `best.pt`
  selection, and the 10 test walks only in `eval.py`.
- Rows: the section-3 usable-row filter. The archive's observation rows were re-derived from the recordings and matched.
- Goals, labels and goal sentences are not used. A world model trains on frames, states and actions; it does not
  train on policy targets.

## Declared settings and deviations

- **Clips:** 8 frames, each 8 raw rows apart (3.75 Hz, the DROID rate), all 8 usable and inside one segment:
  653,784 training clips. Because every frame of a clip must be usable, 2.3% of usable training rows (18,470 of
  800,571) fall in no clip. The loss is spread evenly over motion stages.
- **State and action:** state = measured tool xyz (m), Euler 'xyz' of the measured angle-axis (rad), and gripper
  closedness = clip((0.0340 - jaw stroke) / 0.0031, 0, 1), saturating so every grasp reads 1. Action = difference of
  consecutive states, as in DROID. Commanded poses and jaw intent are not used.
- **Augmentation:** none. 224 px frames are resized to 256 px, the encoder's pretraining grid.
- **Normalization:** none on states or actions (as in the released recipe). Latents are layer-normed without
  learned parameters.
- **Evaluation:** this is a world model, so it does not produce the section-8 action chunks. `eval.py` reports
  held-out teacher-forced and rollout L1 against a copy-last-frame floor, plus a shuffled-action gap, broken down by
  motion stage.

## Plain planning evaluation

`plan_eval.py` plans with the repo's own planner, `notebooks/utils/mpc_utils.py::cem`, at its default settings: 400
samples, top-10, 10 iterations, maxnorm 0.05 m, rotation fixed at 0. It plans from a held-out frame plus its measured
state toward a real goal image H steps later, and compares the plan with the actions actually taken. There are no
subgoals, no task solver, no fixed axes and no step caps. Results are in
`/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/plan/SUMMARY.md`.

```bash
sbatch --export=ALL,CONFIG=<config>,CKPT=<folder>/best.pt,ARGS="--horizon 1 --max_windows 1024" app/vjepa_hanoi/plan_eval.sbatch
```

Long-horizon tests (e.g. the full-stack tasks, 15 moves) should give the model only the final goal image under this
same planner. That is the V-JEPA 2-AC baseline as published.

### Where plain planning works: goal-distance sweep

`goal_distance_eval.py` uses the same default planner and plans from held-out rows inside ring transfers toward a
single goal frame:
- k model steps ahead (`step_1` ... `step_16`, 0.27-4.3 s);
- the end of the current motion stage (`stage_end`);
- the end of the current game move or of 3, 7 or 15 moves (`move_end_1` ... `move_end_15`).

It scores the first planned step against the expert's next step, by stage, on the same rows for every goal type.
Results go to `/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/goal_distance/`.

## Plain baseline on the arm (goal image only)

1. On a GPU node, start the server:
   ```bash
   python -m app.vjepa_hanoi.goal_planner --fname <yaml> --checkpoint <best.pt> --host 0.0.0.0 \
       --goal_archive <openpi play_train.npz>
   ```
   It runs the default repo planner toward one goal image. Execution safety only (applied after planning, reported
   per step): a workspace clamp and a 0.10 m/s speed limit.
2. On the robot host, tunnel with `ssh -L 8766:<gpu node>:8766 <cluster>`, then:
   ```bash
   PYTHONPATH=<tower_hanoi>:<vjepa2> python -m app.vjepa_hanoi.arm_goal_client --start AAAA --goal-board BAAA \
       --log run.npz
   ```
   Alternatively, pass `--goal-image goal.npy`, captured beforehand with `--save-goal-image` after arranging the
   goal board. With `--goal-board`, the goal is one training frame of that board at the end of a move (arm
   retreated). The step budget defaults to 2 x optimal game moves x 120 steps. There is no subgoal, route or
   success logic: check the board afterwards.
3. Rehearse with `--dry-run --replay <recording.h5>`.

### H15 on the robot machine (play or expert model, no cluster)

This is the published baseline: one goal image (the final board), the repo's default planner (horizon 2, 400 samples,
10 CEM iterations), fps 4. It uses no subgoals, no categorical planner and no planner-side constraint. Two bundles are
on the cluster, exported by `robot_results/2026-10-13/cluster_offline/tools/export.sbatch`, with their goal images and
workspace box from the model's own training data:
- `robot_bundle_play`: the official model fine-tuned on the H15 play data (A, `A_vitg_finetune_v1`), 81 goal boards;
- `robot_bundle_expert`: the official model fine-tuned on the six-task expert data (A-expert,
  `A_expert6_vitg_finetune_v1`), 45 goal boards.

Both have AAAA, BBBB and CCCC. `MODEL=cplay` / `MODEL=cexpert` export the predictor-from-scratch models (C) the same
way. A dry run of the expert bundle (server + `arm_goal_client --dry-run`, 3 steps toward CCCC) ran end to end.

1. Copy the bundles (5.3 GB each):
   ```bash
   rsync -avhPr --append-verify \
     cw5167@login.torch.hpc.nyu.edu:/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/robot_bundle_expert/ \
     ~/vjepa_bundle_expert/
   rsync -avhPr --append-verify \
     cw5167@login.torch.hpc.nyu.edu:/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/robot_bundle_play/ \
     ~/vjepa_bundle_play/
   ```
2. Terminal 1, the server (vjepa2's torch environment). Pick the model with `M` (`expert` or `play`). On the 16 GB
   GPU the default planner needs `--predictor_chunk 100`:
   ```bash
   M=expert
   cd /home/cpw/workspace/vjepa2
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m app.vjepa_hanoi.goal_planner \
       --fname ~/vjepa_bundle_$M/config.yaml --checkpoint ~/vjepa_bundle_$M/weights.pt \
       --goal_bundle ~/vjepa_bundle_$M/goals.npz --predictor_chunk 100
   ```
3. Terminal 2, the runner (tower_hanoi's environment, not conda). Example: AAAA → CCCC, a 15-move task:
   ```bash
   conda deactivate
   source /opt/ros/jazzy/setup.bash
   cd /home/cpw/workspace/vjepa2
   M=expert
   EXECUTE=1 PYTHONPATH=/home/cpw/workspace/tower_hanoi:/home/cpw/workspace/vjepa2:$PYTHONPATH \
     uv run --project /home/cpw/workspace/tower_hanoi --extra robot --extra ros \
     python -m app.vjepa_hanoi.arm_goal_client --execute --start AAAA --goal-board CCCC --steps 130 \
     --log runs/h15/${M}_AAAA_CCCC_r1.npz --tower-hanoi /home/cpw/workspace/tower_hanoi
   ```
   - The six H15 tasks are AAAA → BBBB / CCCC, BBBB → AAAA / CCCC and CCCC → AAAA / BBBB.
   - **Time.** The expert takes about 65 model steps (17 s of motion) per game move, so a 15-move task is about 975
     steps. This planner takes about 39 s per step on the RTX 5080 (horizon probe, `robot_results/2026-10-13/RESULTS.md`).
     - `--steps 130` is twice the expert's first move: about 1.4 h.
     - `--steps 975` is the expert's time for the whole task: about 10.5 h.
     - The default budget (no `--steps`) is 2 × 15 moves × 120 steps = 3,600 steps, 3.7 times the expert's time:
       about 39 h.
   - The runner parks at the start pose, then plans and moves until the budget is used. It has no success detector:
     check the board afterwards.
   - Its abort-only checks stay on, as in every arm run, and can end a run early: a planned step more than 3 mm sideways
     with an end below the low floor (166 mm), or a close below that height that grabs nothing. The arm then retreats;
     the steps so far are kept.
   - Every step goes to `--log`, with the server's checkpoint, planner and goal. A log is never overwritten: give each
     run a new name (`_r2`, ...).
   - The runner refuses a server that is not running the plain planner (e.g. one started with `--protocol paper`).

## Six-task expert data

`build_expert_archive.py` rebuilds the Cosmos six-task split into the archive format above:
- source: `cosmos-policy/data/hanoi_cosmos/multitask_v6`, six directed full-stack recordings;
- split: episodes 0-7 train, 8 validate, 9 test;
- rows: every non-stale, non-repeated row, taken verbatim from the Cosmos archives.

`configs/train/vitg16/hanoi-expert6-ac-{ft,scratch}-256px-8f.yaml` train A-expert and C-expert with the play runs'
hyperparameters.

## Why one goal image fails within a game move: energy profile

`energy_profile.py` uses the frozen encoder only, with no predictor and no planner. It measures the L1 energy to a
goal frame along the expert's own path. Toward the end-of-move image, the expert's own descend and insert steps go
uphill (about 70% of steps): the goal shows the arm raised, so moving down moves away from it. The descend barrier
lasts about 9 model steps, beyond a 1-2 step planning horizon. Toward the end of the current motion stage, the path
is mostly downhill. Results: `/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/energy_profile/SUMMARY.md`.

## The paper's pick-and-place protocol: one game move

The V-JEPA 2 paper (arXiv 2506.09985, sec. 4.2) plans pick-and-place as follows:
- two subgoal images before the final goal: the object grasped, then the object near the goal position;
- goals switched on a fixed time schedule;
- CEM at planning horizon 1, with 800 samples and 10 refinement steps, on the L1 energy;
- one action executed before re-planning.

For one ring transfer, the three goals are:
- goal 1: the end of the grasp (ring grasped at the source peg);
- goal 2: the end of the transit (ring above the target peg);
- goal 3: the end of the release (ring on the target peg, arm still at it). `--final_stage 9` uses the end of the
  retreat instead.

Each goal is one frame of a training move of the same board transition. These stand in for the experimenter-provided
images of the paper; on the arm you can pass your own photos instead. The schedule is the median number of expert
steps per goal over the training demos of the same move (all training moves for uploaded photos). Over all moves it is
25 / 23 / 12 steps on the play and the six-task splits. For the six one-move cases it is 29 / 16 / 11 (AAAA>BAAA,
CCCC>BCCC), 29 / 24 / 11 (AAAA>CAAA, CCCC>ACCC) and 26 / 16 / 12 (BBBB>ABBB, BBBB>CBBB).

- **Offline check.** `paper_protocol_eval.py` steps along held-out moves and compares the planner's step toward the
  active goal with the expert's next step.
  - `--switch time` is the paper's fixed schedule; `--switch stage` is perfect switching (an upper bound).
  - `--goals final_only` is the same planner with the final image only.
  - `--goal_source test` uses the held-out move's own frames instead of training frames.
  - Results: `/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/paper_protocol/`.
- **On the arm.** Start the server with `--protocol paper` (horizon 1, 800 samples, 10 iterations), then run:
  ```bash
  PYTHONPATH=<tower_hanoi>:<vjepa2> python -m app.vjepa_hanoi.arm_goal_client --protocol paper --start AAAA \
      --goal-board BAAA --log run_paper_AAAA_BAAA.npz
  ```
  - `--start` and `--goal-board` must be one ring transfer apart.
  - `--goal-images grasped.npy,near.npy,final.npy` uses your own photos.
  - `--schedule 25,23,12` overrides the switch points; the budget is the schedule total.
  - `--fix_x` on the server (off by default) holds x at 0 with the repo planner's own `axis` option, a disclosed
    deviation. It is only meaningful for play-trained models. In the play data x stays on the peg line, so those
    models cannot learn it and plan it at random (a median of 26 mm per step offline). Never use it from the episode
    start pose: in the one-move data the arm starts 8 cm back and moves x from 0.414 to 0.498 m during the approach.
  - `--hold_grasp` on the server (off by default, paper protocol only) holds the gripper closed while goal 2 (ring
    above the target) is active, using the repo planner's own `close_gripper` option. Also a disclosed deviation:
    without it, the planner commands "open" on most carrying steps offline and is expected to drop the ring.

Deviations from the paper:
- The repo planner clips each axis to ±0.05 m where the paper uses an L1 ball of radius 0.075. Neither binds the
  expert (steps at most 3.3 cm), but CEM's plans do sit at the clip in 67-82% of rows
  (`robot_results/2026-10-13/RESULTS.md`).
- The paper's 4 / 10 / 4-step schedule belongs to its own task, so the Hanoi schedule is set from expert timing.

This protocol covers one game move. A 15-move task would need subgoal images for every move, i.e. a task planner,
which the published method does not have.

## Real-robot one-move test (the six first moves, `one_move_test.py`)

**Results so far:** `robot_results/2026-10-13/RESULTS.md` (first runs on the arm: the paper protocol closes the
gripper above the ring at step 1; offline probes of maxnorm, step caps, horizon and simple subgoals; next steps).

The model is A fine-tuned on `data/expert6_first_move`. That is the same 48 training episodes the VLA and
world-action models get: the first move of each H15 task, ring 1 off a full stack. The planning server runs on the
robot machine's GPU; nothing runs on the cluster.

1. **Export a bundle** (once, on the cluster, CPU only):
   ```bash
   python -m app.vjepa_hanoi.export_robot_bundle --fname configs/train/vitg16/hanoi-firstmove-ac-ft-256px-8f.yaml \
       --checkpoint <A_firstmove_vitg_finetune_v1/best.pt> --goal_archive <data/expert6_first_move/train.npz> --out <bundle>
   ```
   It writes:
   - `weights.pt` (5.3 GB): the frozen encoder and the predictor, float32, with no optimizer state;
   - `goals.npz` (1.4 MB): each move's three goal frames and its schedule, plus the workspace box;
   - `config.yaml`;
   - `MANIFEST.json` with sha256 hashes.

   Copy the folder to the robot machine.
2. **Robot machine, terminal 1: the server.** It needs a CUDA GPU and this repo's Python dependencies: torch,
   torchvision, timm, einops, numpy, scipy, PyYAML, h5py, opencv-python-headless, pillow.
   ```bash
   python -m app.vjepa_hanoi.goal_planner --fname <bundle>/config.yaml --checkpoint <bundle>/weights.pt \
       --goal_bundle <bundle>/goals.npz --protocol paper --host 127.0.0.1 --port 8766
   ```
   `curl -s http://127.0.0.1:8766/health` shows the checkpoint, protocol and safety box. On a 16 GB GPU (RTX 5080) one
   batch of 800 CEM samples runs out of memory: add `--predictor_chunk 200` and run with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (same plans, 12.4 GiB peak, about 25 s per step).
3. **Robot machine, terminal 2: the runner.** It runs in tower_hanoi's environment (numpy + tower_hanoi; no torch).
   The real arm moves only with `--execute` plus `EXECUTE=1` (tower_hanoi's hardware rule); a dry run takes neither.
   Keep `$PYTHONPATH`: it carries ROS's rclpy for the camera.
   ```bash
   EXECUTE=1 PYTHONPATH=<tower_hanoi>:<vjepa2>:$PYTHONPATH python -m app.vjepa_hanoi.one_move_test --execute \
       --out runs/one_move_<date> --tower-hanoi <tower_hanoi>
   ```
   - It runs the six cases × `faithful,hold` (12 runs), prompting you to set the board before each run and asking
     afterwards how far the move got (0–5, or x = invalid, redo). Only a placement (5) asks whether another ring
     moved, and the gripper questions come only when the jaw reads closed.
   - `--resume` continues a session.
   - `--repeats N` repeats every case and variant.
   - Each run's log has every step with its camera frame; `<run>.steps.jsonl` gets one line per step as the run goes
     (a killed process keeps it); one row per attempt goes to `results.jsonl`. The session file records the git HEAD,
     the files that differ from it and a hash of that difference.
   - `app/vjepa_hanoi/serve.sbatch` runs the same server on a cluster GPU instead, reached over an ssh tunnel.

**The setup that works offline** (`robot_results/2026-10-13/cluster_offline/RESULTS_2026-10-15.md`: 6/6 pick, 5/6
whole move in the replay). It uses a different bundle and opt-in server flags, and it is not the paper baseline. The
bundle is `robot_bundle_efps1`: E-1fps, 1 s steps, trained on all expert moves of the six tasks. The flags are:
- manual stage subgoals;
- each step is the best of 51 training actions;
- no sideways step near the rods (the floor must equal the arm's low floor, or the runner refuses before any motion);
- no step out of the box;
- steps capped at 36 mm, schedule ×1.5.
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m app.vjepa_hanoi.goal_planner --fname <bundle>/config.yaml \
    --checkpoint <bundle>/weights.pt --goal_bundle <bundle>/goals.npz --protocol paper --subgoals stages \
    --sampling cat --guard_floor_mm 166.13 --box_feasible --max_speed 0.036 --predictor_chunk 200 \
    --host 127.0.0.1 --port 8766
EXECUTE=1 PYTHONPATH=<tower_hanoi>:<vjepa2>:$PYTHONPATH python -m app.vjepa_hanoi.one_move_test --execute \
    --variants stages --schedule_scale 1.5 --out runs/one_move_efps1_<date> --tower-hanoi <tower_hanoi>
```
Each step scores 51 actions, not 800 × 10 CEM samples. `--predictor_chunk 200` changes nothing at that size; it is
there only for the 16 GB GPU.

Before the arm moves, the runner refuses to start unless the robot host's tower_hanoi config matches the recordings:
- the gripper opens to 0.034 (`open_fraction` 0.85 in `workspace.json`);
- `wm-transform.json` is square_roi 151/90/360 → 224;
- the open jaw reading is at least 0.033.

Each run starts, by default (`--start hover`, as tower_hanoi's Q-WM runner), above the source peg's calibrated grasp
point at the start height (for peg A (498.4, -57.2, 191.1) mm), reached by going straight up, then one leg above the
rods. The demos pass there at the end of their approach, 15.9 model steps in for source pegs A and C and 13.2 for B,
jaw open and nearly still, so goal 1's step budget is shortened by that much: the schedule becomes the demos' own
timing from the hover, [13, 16, 11] or [13, 24, 11]. `--start episode` starts at the demos' first pose
(0.4140, 0.0157, 0.1911), 8 cm behind the rods, with the full schedule.

Server-side safety:
- every step is clamped to the workspace box of the training data (never more than 1 mm below its lowest tool height),
  then limited to 0.10 m/s;
- the orientation is always the recorded one;
- a pose outside the box is refused.

Execution on the arm follows tower_hanoi's Q-WM runner (`tower_hanoi/docs/handover-hardware-test-pipeline.md`). It
changes how the arm reaches the planned target, never the target, and the schedule still counts model steps:
- each move's goal time is Q-WM's rule instead of the data's 0.267 s: distance / 36 mm/s within 0.5-1.0 s, and 1.0 s
  for a jaw change, the 2 steps after one and a loaded step below the low floor (transit height - 25 mm, 166.1 mm);
- the 3 mm / 2 deg readback re-reads the pose for up to 0.15 s before it fails (same tolerances);
- the planner's frame is one received after the arm is still (<= 0.02 m/s on 3 readings 10 ms apart);
- abort-only checks, which never steer: tower_hanoi's missed-grasp check after a close below the low floor, where a
  ring or a rod can be between the pads (a missed grasp there ends the run; a close higher up, e.g. in mid-air at the
  start pose, is only logged and the planner may reopen), and no sideways step over 3 mm with an end below the low
  floor (Q-WM's guard without its recovery).
- before a run moves anything, a camera frame must arrive. rclpy's own SIGINT handler is off, so a Ctrl-C (during a
  run or at a prompt) no longer stops the ROS camera.

Ctrl-C stops a run with no further motion, then asks before the retreat. The retreat (end of a run, an abort, the
session end) opens the jaw first when the tool is below the low floor, so a ring stays on its rod, then rises straight
up to the start height; a ring held higher stays held. If that opened a jaw that was not open, the runner released the
ring, not the model: the run records `runner_released_jaw` and cannot count as a success. After a run without errors
the arm moves at the start height to the start pose, 8 cm behind the rods, for the board reset. On exit tower_hanoi
parks the arm, but if the retreat failed it parks only when the operator types PARK; otherwise the process waits, with
the arm holding, until the operator has recovered it (the controller goes idle when the process exits). The e-stop
remains the real safety.

Task-specific planning aids (a symbolic route solver, per-stage subgoal images, holding x fixed, a step cap) are kept
separately in `app/vjepa_hanoi_oracle/`. They use privileged information and are not part of this baseline.


## H15 with the one-move setup (`h15_test.py`)

This plays each H15 task as its 15 optimal moves. Every move runs exactly as one `one_move_test` run with the setup
that works on one move (stage subgoals, 51 training actions, the guard, the box, 36 mm steps, schedule ×1.5).
- The server sets that move's six stage goals (from the first training demo of the same transition) and its schedule
  (that transition's training medians, counted from the hover).
- The arm opens the jaw, parks above the source peg at the start height, runs the schedule, and retreats.
- The hover is the same for every H15 move: the 90 held-out demos pass within 1 mm of it at the end of their approach.

So each move differs from an H1 run only in the board the previous moves left. This is not the paper baseline: it
gives the planner 15 × 6 manual subgoal images.

**Bundle `robot_bundle_efps1h15`.** It holds the same weights as `robot_bundle_efps1` (E-1fps), with goals for all 90
H15 moves from `data/expert6_single_move/train.npz`. Two things must come from all moves, not first moves:
- **The box.** The first moves' box stops at 85.7 mm, but later moves grasp and place down to 56.5 mm.
- **The 51 actions.** With the first moves' set, later moves fail offline: 8/84.

Export it with `sbatch --export=ALL,MODEL=efps1h15 tools/export.sbatch`; MANIFEST sha256 of weights.pt 3987bca2….

**Offline, all 90 held-out H15 moves** (`descent_replay.py --until release`, closed loop on recorded frames; the arm
setting is guard + box + 36 mm cap + schedule ×1.5; `tools/h15_replay_report.py`):

| | all 90 moves | the 6 first moves | later moves |
|---|---|---|---|
| arm setting, success as for H1 | 48/90 | 3/6 (`robot_bundle_efps1`: 5/6) | 45/84 |
| same, counting a release over the rod as placed | 82/90 | | |
| full 1 s steps, no cap (not allowed on the arm today) | 67/90 | 5/6 | 62/84 |

- "As for H1" means the jaw opens within 10 mm (5 mm sideways) of the demo's release pose. "Over the rod" counts an
  opening up to 15 mm higher, still within 5 mm sideways and below the low floor. 34 of the 42 misses are of that kind,
  so whether the ring slides down the rod decides most of them.
- Later moves do about as well as first moves.
- Ring 4 (move 8) is 3/6 with the arm setting and 0/6 at full steps: the jaw stays open 5 mm above it.

On the robot machine (`/home/cpw/workspace/vjepa2` at this commit; copy the bundle once):
```bash
rsync -avhPr --append-verify \
  cw5167@login.torch.hpc.nyu.edu:/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/robot_bundle_efps1h15/ \
  /home/cpw/vjepa_bundle_efps1h15/
```
Terminal 1 (the vjepa2 torch environment):
```bash
cd /home/cpw/workspace/vjepa2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m app.vjepa_hanoi.goal_planner \
    --fname /home/cpw/vjepa_bundle_efps1h15/config.yaml --checkpoint /home/cpw/vjepa_bundle_efps1h15/weights.pt \
    --goal_bundle /home/cpw/vjepa_bundle_efps1h15/goals.npz --protocol paper --subgoals stages --sampling cat \
    --guard_floor_mm 166.13 --box_feasible --max_speed 0.036 --predictor_chunk 200 --host 127.0.0.1 --port 8766
```
Terminal 2 (tower_hanoi's environment):
```bash
conda deactivate; source /opt/ros/jazzy/setup.bash; cd /home/cpw/workspace/vjepa2
EXECUTE=1 PYTHONPATH=/home/cpw/workspace/tower_hanoi:/home/cpw/workspace/vjepa2:$PYTHONPATH \
  uv run --project /home/cpw/workspace/tower_hanoi --extra robot --extra ros \
  python -m app.vjepa_hanoi.h15_test --execute --tasks 'AAAA>CCCC' --variants stages --schedule_scale 1.5 \
  --out runs/h15_efps1h15_2026-10-16 --tower-hanoi /home/cpw/workspace/tower_hanoi
```
- **Before any motion.** The runner checks the server: paper protocol, steps ≤ 36 mm, and the guard floor equal to this
  arm's. It also asks the server for the goals and schedule of every move it will run. A one-move bundle such as
  `robot_bundle_efps1` is refused here.
- **After every move.** You answer how far it got (0–5, x = invalid) and, for a placement, whether another ring moved.
  A success moves on to the next move. After a failure:
  - `c` sets the board by hand to the result the move should have made, then continues;
  - `r` sets it back and redoes the move;
  - `q` ends the task.

  For `c` and `r` the arm first goes to the start pose behind the rods. Both are recorded as interventions.
- **If the retreat is declined or fails**, the arm does not move again: the session ends with its exit retreat (the jaw
  opens first below the low floor, PARK only if that fails). Continue with `--append --from-move N`.
- **Results.** `results.jsonl` gets one row per move attempt (task, move, ring, pegs, progress, success,
  after_intervention). `tasks.jsonl` gets one row per task: moves before the first failure, strict success (all 15
  from move 1 with no intervention), and the attempts. The per-move logs (`<task>_stages_r0_mNN.npz` / `.steps.jsonl`)
  are like `one_move_test`'s.
- `--tasks` takes any of the six (`AAAA>BBBB AAAA>CCCC BBBB>AAAA BBBB>CCCC CCCC>AAAA CCCC>BBBB`, comma-separated).
- **Cluster rehearsal** (dry run, the cluster's floor 175.32): `tools/rehearse.sbatch` with `CLIENT=h15`.

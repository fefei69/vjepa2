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
images of the paper; on the arm you can pass your own photos instead. The schedule is the training split's median
number of expert steps per goal: 25 / 23 / 12 steps on both the play and the expert training splits.

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
  - `--fix_x` on the server (off by default) holds x at 0 with the repo planner's own `axis` option. This is a
    disclosed deviation. x never varies in this data, so the model cannot learn what it does, and the planned x is
    otherwise random (a median of 26 mm per step offline). Running with and without it separates x drift from the
    protocol itself.

Deviations from the paper:
- The repo planner clips each axis to ±0.05 m where the paper uses an L1 ball of radius 0.075. Neither binds here:
  expert steps are at most 3.3 cm.
- The paper's 4 / 10 / 4-step schedule belongs to its own task, so the Hanoi schedule is set from expert timing.

This protocol covers one game move. A 15-move task would need subgoal images for every move, i.e. a task planner,
which the published method does not have.

Task-specific planning aids (a symbolic route solver, per-stage subgoal images, holding x fixed, a step cap) are kept
separately in `app/vjepa_hanoi_oracle/`. They use privileged information and are not part of this baseline.


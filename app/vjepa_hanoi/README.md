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

Task-specific planning aids (a symbolic route solver, per-stage subgoal images, holding x fixed, a step cap) are kept
separately in `app/vjepa_hanoi_oracle/`. They use privileged information and are not part of this baseline.


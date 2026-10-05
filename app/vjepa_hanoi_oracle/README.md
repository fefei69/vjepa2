# Oracle-subgoal planning with V-JEPA 2-AC (NOT the baseline)

This folder drives a trained V-JEPA 2-AC checkpoint (from `app/vjepa_hanoi`) on the Hanoi task with **privileged
help** that the plain world model does not have:

- a **symbolic solver** (`arm_client.route`, breadth-first search over the 81 boards) that decides the sequence of
  ring moves;
- **seven subgoal images per move**, one at the end of each motion stage, taken from expert training demos of the
  same board transition (`subgoals.py`);
- **x held fixed** (x never moves in the training data) and a **20 mm per-axis step cap**.

Results with these aids are an **upper bound**: they show the model can execute single motions when an external
discrete planner supplies the plan. Do not report them as the V-JEPA 2-AC baseline. The baseline (training, test
evaluation, plain planning with the repo planner and a goal image) is in `app/vjepa_hanoi/`.

## Pieces

1. `subgoals.py` builds the subgoal library from the training split only: 1,680 (transition, phase) subgoals covering
   all 240 transitions. The built file is
   `/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/oracle_subgoal_planner/subgoals_train.npz`.
   ```bash
   python -m app.vjepa_hanoi_oracle.subgoals --archive <openpi play_train.npz> --out subgoals.npz
   ```
2. `planner.py` runs the repo CEM with the aids above, one 0.267 s step per request, over HTTP.
   `validate` runs the same planner offline on held-out walks.
   ```bash
   python -m app.vjepa_hanoi_oracle.planner serve --fname configs/train/vitg16/hanoi-play-k5-ac-ft-256px-8f.yaml \
       --checkpoint <A best.pt> --subgoals <subgoals.npz> --host 0.0.0.0
   ```
3. `arm_client.py` runs on the robot host, using tower_hanoi's `TrossenArm`, `RosCamera` and `wm-transform.json`.
   It stops for the operator if a subgoal is not reached. Rehearse with `--dry-run --replay <recording.h5>`.
   ```bash
   PYTHONPATH=<tower_hanoi>:<vjepa2> python -m app.vjepa_hanoi_oracle.arm_client --start AAAA --goal CCCC
   ```

## Offline check (run A, 1,000 test-walk rows; `oracle_subgoal_planner/A_subgoal_steering_test.json`)

The planner's next step is compared with the expert's next step:

- direction is correct on 93% of moving rows;
- the gripper closes and opens at the right moments in 92% and 100% of cases;
- spurious motion while the arm should hold still is 5.8 mm.

A first version with only three subgoals per move went the wrong way while descending and inserting. Its results are
kept as `*_v1_3phase*`.

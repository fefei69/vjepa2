# More first-move data for the H1 model (2026-10-17)

**Question.** The models trained on the 48 first-move demos alone overfit within about 6 epochs and fail offline in
the arm setting (A-1fps: 0/6). Does more data of the same six moves fix this, without adding any other move?

**Data** (`app/vjepa_hanoi/build_first_move_plus_archive.py`). The six first moves exist in only three more places:
the Sept 10 and Sept 15 expert AAAA↔CCCC sessions (100 episodes each, move 0 = AAAA>BAAA or CCCC>BCCC) and the play
recording (91 of its moves). The rule that picks usable frames rebuilds the original 48-demo split exactly. Sept 10 has
no stage, stale or repeated columns; they were rebuilt and checked against Sept 15 (header of the builder).

| Archive | Demos per move (AAAA>BAAA, AAAA>CAAA, BBBB>ABBB, BBBB>CBBB, CCCC>ACCC, CCCC>BCCC) | Demos | fps-1 clips |
|---|---|---|---|
| expert6_first_move (A-1fps) | 8 each | 48 | 11,940 |
| first_move_plus_expert (A-firstmoveplus-1fps) | 108, 8, 8, 8, 8, 108 | 248 | 51,928 |
| first_move_plus_all (A-firstmoveplusall-1fps, + play) | 116, 16, 16, 15, 39, 137 | 339 | 75,236 |
| H15 expert, all moves (E-1fps, for reference) | (every move of the six tasks) | 720 moves | 282,839 |

**Training.** A-1fps's recipe with E-1fps's budget (25 epochs × 300 steps × 32 clips), so the new runs differ from
E-1fps only in data. Validation and test are the first-move split (episodes 8 and 9 of the six-task files), as before.
Jobs 20155779 / 20156098 (2 h 31 min / 2 h 33 min), exports 20156099 / 20156100, replay 20156101; commit 5559389.

| Model | Validation loss, best (epoch) | At the last epoch |
|---|---|---|
| A-1fps (48 demos, 15 epochs) | 0.5996 (6) | 0.6309 (rising from epoch 7: overfits) |
| A-firstmoveplus-1fps (248 demos) | 0.5885 (21) | 0.5899 |
| A-firstmoveplusall-1fps (339 demos) | **0.5793 (25)** | 0.5793 (still falling) |
| E-1fps (all H15 expert moves) | 0.5876 (25) | 0.5876 |

The overfitting is gone. The play-augmented model now beats E-1fps on held-out first moves.

**Offline whole move** (`tools/descent_replay.py --until release`, the six held-out first moves, closed loop on
recorded frames, goals/codebook/box from the first-move train split for every model; `replay/*_replay_release_sched.json`):

| Model | Arm setting: guard + box + 36 mm cap + schedule ×1.5 | Full 1 s steps (guard + box) |
|---|---|---|
| A-1fps | 0/6 | 1/6 |
| **A-firstmoveplus-1fps** (`robot_bundle_h1plus`) | **4/6** | 5/6 |
| A-firstmoveplusall-1fps (`robot_bundle_h1plusall`) | 4/6 | 4/6 |
| E-1fps (`robot_bundle_efps1`, earlier runs) | 5/6 | 4/6 |

- In the arm setting, A-firstmoveplus-1fps succeeds on AAAA>BAAA, AAAA>CAAA, BBBB>CBBB and BBBB>ABBB. Three of these
  four gained no demos, so the extra data helps the shared skill, not just the two moves it covers. Both misses
  (CCCC>BCCC, CCCC>ACCC) close the jaw 10.7 mm above the demo's grasp height, just outside the 10 mm grasp window
  (the replay calls this "reopened"). This is the missed-grasp failure the arm showed with A-1fps.
- The bundles hold the same `goals.npz` as `robot_bundle_fps1` (sha256 6308669e…), so on the arm only the weights
  change. weights.pt sha256: h1plus 4ce48ca1…, h1plusall 3597387e….

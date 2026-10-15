#!/bin/bash
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --job-name=vjepa2-hanoi-export
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/slurm/%x-%j.out

# Robot bundle (weights.pt, goals.npz with the action codebook and board goals, config.yaml, MANIFEST.json), on CPU:
#   sbatch --export=ALL,MODEL=efps1 <this file>      -> $K/robot_bundle_<MODEL>
# MODEL as offline.sbatch: fps1 (first moves, fps 1), efps1 (all expert moves, fps 1), efps2 (all expert moves, fps 2),
# with goals from the first-move train split; or the fps-4 models of the H15 baseline, with goals (and box) from their
# own training split: play (A, play_k5), cplay (C, play_k5), expert (A-expert, six tasks), cexpert (C-expert).
set -euo pipefail
REPO=/scratch/cw5167/workspace/vjepa2
K=/scratch/cw5167/checkpoints/vjepa2_ac_hanoi
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
GOALS="$K/data/expert6_first_move/train.npz"
PLAY=/scratch/cw5167/workspace/openpi/data/hanoi/play_k5_pi05/indices/play_train.npz
EXPERT="$K/data/expert6_multitask_v6/train.npz"
case "${MODEL:?set MODEL}" in
  fps1) CFG=configs/train/vitg16/hanoi-firstmove-ac-ft-256px-8f-1fps.yaml; RUN=A_firstmove_1fps_vitg_finetune_v1 ;;
  efps1) CFG=configs/train/vitg16/hanoi-expert6all-ac-ft-256px-8f-1fps.yaml; RUN=A_expert6all_1fps_vitg_finetune_v1 ;;
  efps2) CFG=configs/train/vitg16/hanoi-expert6all-ac-ft-256px-8f-2fps.yaml; RUN=A_expert6all_2fps_vitg_finetune_v1 ;;
  play) CFG=configs/train/vitg16/hanoi-play-k5-ac-ft-256px-8f.yaml; RUN=A_vitg_finetune_v1; GOALS=$PLAY ;;
  cplay) CFG=configs/train/vitg16/hanoi-play-k5-ac-scratch-256px-8f.yaml; RUN=C_vitg_scratch_v1; GOALS=$PLAY ;;
  expert) CFG=configs/train/vitg16/hanoi-expert6-ac-ft-256px-8f.yaml; RUN=A_expert6_vitg_finetune_v1; GOALS=$EXPERT ;;
  cexpert) CFG=configs/train/vitg16/hanoi-expert6-ac-scratch-256px-8f.yaml; RUN=C_expert6_vitg_scratch_v1; GOALS=$EXPERT ;;
  *) echo "unknown MODEL $MODEL" >&2; exit 2 ;;
esac
echo "job $SLURM_JOB_ID on $(hostname -s), commit $(git rev-parse --short HEAD)"
"$REPO/.venv/bin/python" -m app.vjepa_hanoi.export_robot_bundle --fname "$CFG" --checkpoint "$K/$RUN/best.pt" \
  --goal_archive "$GOALS" --out "$K/robot_bundle_${MODEL}" 2>&1 | grep -v Warning
ls -la "$K/robot_bundle_${MODEL}"

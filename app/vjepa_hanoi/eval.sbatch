#!/bin/bash
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --job-name=vjepa2-ac-hanoi-eval
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/slurm/%x-%j.out

# Held-out evaluation of a checkpoint written by app.vjepa_hanoi.train (writes <CKPT>.<SPLIT>_eval.json).
#   sbatch --export=ALL,CONFIG=<config yaml>,CKPT=<folder>/best.pt[,SPLIT=test][,EXTRA="--max_windows 256"] \
#          app/vjepa_hanoi/eval.sbatch

set -euo pipefail

REPO=/scratch/cw5167/workspace/vjepa2
CONFIG="${CONFIG:?set CONFIG=<config yaml>}"
CKPT="${CKPT:?set CKPT=<checkpoint .pt>}"
SPLIT="${SPLIT:-test}"
cd "$REPO"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1

echo "job $SLURM_JOB_ID on $(hostname): config=$CONFIG ckpt=$CKPT split=$SPLIT extra=${EXTRA:-}"
# shellcheck disable=SC2086
"$REPO/.venv/bin/python" -m app.vjepa_hanoi.eval --fname "$CONFIG" --checkpoint "$CKPT" --split "$SPLIT" \
    --num_workers 6 ${EXTRA:-}

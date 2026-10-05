#!/bin/bash
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --job-name=vjepa2-ac-hanoi-plan
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/slurm/%x-%j.out

# Action recovery by planning (app/vjepa_hanoi/plan_eval.py); writes <CKPT>.<SPLIT>_plan_h<H>_<GOAL>.json (+ _clips.npz).
#   sbatch --export=ALL,CONFIG=<config yaml>,CKPT=<ckpt>[,ARGS="--horizon 1 --max_windows 1024"] app/vjepa_hanoi/plan_eval.sbatch

set -euo pipefail

REPO=/scratch/cw5167/workspace/vjepa2
CONFIG="${CONFIG:?set CONFIG=<config yaml>}"
CKPT="${CKPT:?set CKPT=<checkpoint .pt>}"
cd "$REPO"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=4

echo "job $SLURM_JOB_ID on $(hostname): config=$CONFIG ckpt=$CKPT args=${ARGS:-}"
# shellcheck disable=SC2086
"$REPO/.venv/bin/python" -m app.vjepa_hanoi.plan_eval --fname "$CONFIG" --checkpoint "$CKPT" ${ARGS:-}

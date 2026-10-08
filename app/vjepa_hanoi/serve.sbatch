#!/bin/bash
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --job-name=vjepa2-ac-hanoi-serve
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/slurm/%x-%j.out

# The V-JEPA 2-AC planning server for a real-robot session (app/vjepa_hanoi/goal_planner.py), on one GPU for --time.
# Default: the one-move model (A fine-tuned on data/expert6_first_move, the dataset the VLA / world-action models
# train on), the paper's pick-and-place protocol, goal images and schedule from that dataset's training demos.
#   sbatch app/vjepa_hanoi/serve.sbatch
#   sbatch --export=ALL,CKPT=<best.pt>,CONFIG=<yaml>,GOALS=<train.npz>,PORT=8766 app/vjepa_hanoi/serve.sbatch
# Then `grep -A3 TUNNEL <log>` prints the ssh command to run on the robot host. Variant options (fix_x, hold_grasp)
# are chosen per run by the client, so one server covers a whole session.

set -euo pipefail

REPO=/scratch/cw5167/workspace/vjepa2
K=/scratch/cw5167/checkpoints/vjepa2_ac_hanoi
CONFIG="${CONFIG:-configs/train/vitg16/hanoi-firstmove-ac-ft-256px-8f.yaml}"
CKPT="${CKPT:-$K/A_firstmove_vitg_finetune_v1/best.pt}"
GOALS="${GOALS:-$K/data/expert6_first_move/train.npz}"
PORT="${PORT:-8766}"
cd "$REPO"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=4

NODE=$(hostname -s)
echo "job $SLURM_JOB_ID on $NODE: config=$CONFIG ckpt=$CKPT goals=$GOALS port=$PORT commit=$(git rev-parse --short HEAD)"
echo "TUNNEL (run on the robot host, keep it open):"
echo "  ssh -N -L $PORT:$NODE:$PORT <your cluster login>"
echo "  then: curl -s http://127.0.0.1:$PORT/health"
exec "$REPO/.venv/bin/python" -m app.vjepa_hanoi.goal_planner --fname "$CONFIG" --checkpoint "$CKPT" \
    --goal_archive "$GOALS" --protocol paper --host 0.0.0.0 --port "$PORT"

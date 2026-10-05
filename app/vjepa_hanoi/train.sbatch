#!/bin/bash
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --job-name=vjepa2-ac-hanoi
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/slurm/%x-%j.out

# V-JEPA 2-AC on the Hanoi play_k5 split, one task per GPU. Resubmitting the same CONFIG resumes from folder/latest.pt.
#   run A (ViT-g, fine-tune) : sbatch --export=ALL,CONFIG=configs/train/vitg16/hanoi-play-k5-ac-ft-256px-8f.yaml \
#                                     app/vjepa_hanoi/train.sbatch
#   run B (ViT-L, scratch)   : sbatch --export=ALL,CONFIG=configs/train/vitl16/hanoi-play-k5-ac-scratch-256px-8f.yaml \
#                                     app/vjepa_hanoi/train.sbatch
#   smoke (2 GPUs)           : sbatch --ntasks-per-node=2 --gres=gpu:h200:2 --mem=128G --time=00:45:00 \
#                                     --export=ALL,CONFIG=configs/train/<...>-smoke.yaml app/vjepa_hanoi/train.sbatch
# --ntasks-per-node must equal the number of GPUs. 2x H200 by default: 4-GPU requests queue on the h200_tandon /
# h200_public group GPU caps (112 / 24 GPUs); h200 and h100 have no cap but are preemptible (PreemptMode=REQUEUE).

set -euo pipefail

REPO=/scratch/cw5167/workspace/vjepa2
CONFIG="${CONFIG:?set CONFIG=<config yaml>, see the usage above}"
cd "$REPO"

export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"

echo "job $SLURM_JOB_ID on $(hostname): config=$CONFIG tasks=$SLURM_NTASKS gpus=${CUDA_VISIBLE_DEVICES:-none}"
srun --kill-on-bad-exit=1 "$REPO/.venv/bin/python" -m app.vjepa_hanoi.run --fname "$CONFIG"

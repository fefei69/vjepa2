#!/bin/bash
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --job-name=vjepa2-hanoi-descent
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/slurm/%x-%j.out

# Offline descend/grasp experiments on one H200 (no robot). Results land in robot_results/2026-10-13/cluster_offline.
#   sbatch --export=ALL,TASK=probes <this file>          probe_planner (horizon, simple; 800 samples, H 1-3) + hover_probe
#   sbatch --export=ALL,TASK=ppe,SETTINGS="base_h1 stages_h1" <this file>
#                                                        paper_protocol_eval on the six held-out first moves, one run per
#                                                        named setting (see setting() below); ROWS=descent (default:
#                                                        rows in stages 3-4 only) or ROWS=all (whole move)
set -uo pipefail  # one failing setting does not stop the others
REPO=/scratch/cw5167/workspace/vjepa2
K=/scratch/cw5167/checkpoints/vjepa2_ac_hanoi
R=$REPO/app/vjepa_hanoi/robot_results/2026-10-13
OUT=$R/cluster_offline
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
PY="$REPO/.venv/bin/python"
mkdir -p "$OUT/probes" "$OUT/ppe"
echo "job $SLURM_JOB_ID on $(hostname -s), commit $(git rev-parse --short HEAD), dirty: $(git status --porcelain | wc -l) files"

setting() {  # name -> paper_protocol_eval flags; label: B = paper baseline, P = planner deviation, A = task aid
  case "$1" in
    base_h1) echo "" ;;
    base_h2) echo "--rollout 2" ;;
    base_h3) echo "--rollout 3" ;;
    g03_h1) echo "--gripper_std 0.3" ;;
    g03_h2) echo "--gripper_std 0.3 --rollout 2" ;;
    g03_h3) echo "--gripper_std 0.3 --rollout 3" ;;
    g01_h1) echo "--gripper_std 0.1" ;;
    stages_h1) echo "--goals stages" ;;
    stages_h2) echo "--goals stages --rollout 2" ;;
    stages_h3) echo "--goals stages --rollout 3" ;;
    stages_g03_h1) echo "--goals stages --gripper_std 0.3" ;;
    stages_g03_h2) echo "--goals stages --gripper_std 0.3 --rollout 2" ;;
    stages_g03_h3) echo "--goals stages --gripper_std 0.3 --rollout 3" ;;
    aid_xy_h1) echo "--hold_xy" ;;
    aid_stages_open_h1) echo "--goals stages --hold_open" ;;
    aid_stages_xy_h1) echo "--goals stages --hold_xy" ;;
    aid_stages_open_xy_h1) echo "--goals stages --hold_open --hold_xy" ;;
    aid_stages_open_xy_h2) echo "--goals stages --hold_open --hold_xy --rollout 2" ;;
    aid_stages_open_xy_h3) echo "--goals stages --hold_open --hold_xy --rollout 3" ;;
    aid_stages_all_h1) echo "--goals stages --hold_open --hold_xy --hold_grasp" ;;
    aid_stages_all_h2) echo "--goals stages --hold_open --hold_xy --hold_grasp --rollout 2" ;;
    aid_stages_xy_h3) echo "--goals stages --hold_xy --rollout 3" ;;
    aid_stages_xy_grasp_h3) echo "--goals stages --hold_xy --hold_grasp --rollout 3" ;;
    *) echo "unknown setting $1" >&2; exit 2 ;;
  esac
}

if [ "${TASK}" = probes ]; then
  for mode in horizon simple; do
    "$PY" "$R/tools/probe_planner.py" --mode "$mode" --samples 800 --horizons 1,2,3 \
      2>&1 | grep -v Warning | tee "$OUT/probes/${mode}_probe_800.txt"
  done
  "$PY" "$OUT/tools/hover_probe.py" --out "$OUT/probes/hover_probe.jsonl" 2>&1 | grep -v Warning \
    | tee "$OUT/probes/hover_probe.txt"
elif [ "${TASK}" = ppe ]; then
  EVAL="--eval_stages 3,4"; TAG=descent
  if [ "${ROWS:-descent}" = all ]; then EVAL=""; TAG=all; fi
  for s in ${SETTINGS}; do
    flags=$(setting "$s")
    echo "=== $s: $flags $EVAL"
    "$PY" -m app.vjepa_hanoi.paper_protocol_eval --fname configs/train/vitg16/hanoi-firstmove-ac-ft-256px-8f.yaml \
      --checkpoint $K/A_firstmove_vitg_finetune_v1/best.pt --archive $K/data/expert6_first_move/test.npz \
      --goal_archive $K/data/expert6_first_move/train.npz $EVAL $flags --out "$OUT/ppe/${TAG}_${s}.json" \
      2>&1 | grep -v Warning
  done
fi

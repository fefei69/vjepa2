#!/bin/bash
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --job-name=vjepa2-hanoi-rehearsal
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/cw5167/checkpoints/vjepa2_ac_hanoi/slurm/%x-%j.out

# Dry-run rehearsal of the arm session (no robot): the planning server from a bundle with the session's flags, and
# one_move_test --dry-run against it over HTTP (tower_hanoi's DryRunArm; the camera replays the held-out demo, so this
# checks the plumbing: the client's server checks, goals, schedule, every request and reply; not the closed loop).
#   sbatch --export=ALL,BUNDLE=$K/robot_bundle_efps1,SERVER_FLAGS="--subgoals stages --sampling cat ...",\
#          RUNNER_FLAGS="--variants stages --schedule_scale 1.5" <this file>
set -uo pipefail
REPO=/scratch/cw5167/workspace/vjepa2
K=/scratch/cw5167/checkpoints/vjepa2_ac_hanoi
TH=/scratch/cw5167/workspace/tower_hanoi
cd "$REPO"
export OMP_NUM_THREADS=4
PY="$REPO/.venv/bin/python"
B="${BUNDLE:?set BUNDLE}"
OUT="$K/robot_rehearsal/session_${SLURM_JOB_ID}"
mkdir -p "$OUT"
echo "job $SLURM_JOB_ID on $(hostname -s), commit $(git rev-parse --short HEAD); server flags: ${SERVER_FLAGS:-}; runner flags: ${RUNNER_FLAGS:-}"
PYTHONPATH="$REPO" "$PY" -m app.vjepa_hanoi.goal_planner --fname "$B/config.yaml" --checkpoint "$B/weights.pt" \
  --goal_bundle "$B/goals.npz" --protocol paper ${SERVER_FLAGS:-} --host 127.0.0.1 --port 8766 > "$OUT/server.log" 2>&1 &
SERVER=$!
for i in $(seq 1 120); do curl -s http://127.0.0.1:8766/health > "$OUT/health.json" 2>/dev/null && break; sleep 5; done
echo "health: $(head -c 600 "$OUT/health.json")"
PYTHONPATH="$TH:$REPO" "$PY" -m app.vjepa_hanoi.one_move_test --dry-run --replay-archive "$K/data/expert6_first_move/test.npz" \
  --yes --tower-hanoi "$TH" --out "$OUT/runs" ${RUNNER_FLAGS:-} 2>&1 | grep -v Warning
echo "runner exit ${PIPESTATUS[0]}"
kill "$SERVER"
ls -la "$OUT/runs" 2>/dev/null | head -20

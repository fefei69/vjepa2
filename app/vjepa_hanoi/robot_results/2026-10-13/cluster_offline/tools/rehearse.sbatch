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
# CLIENT=goal runs arm_goal_client instead (the plain goal-image baseline, e.g. H15), with PROTOCOL=plain:
#   sbatch --export=ALL,BUNDLE=$K/robot_bundle_expert,PROTOCOL=plain,CLIENT=goal,SERVER_FLAGS="--predictor_chunk 100",\
#          RUNNER_FLAGS="--start AAAA --goal-board CCCC --steps 3" <this file>
set -uo pipefail
REPO=/scratch/cw5167/workspace/vjepa2
K=/scratch/cw5167/checkpoints/vjepa2_ac_hanoi
TH=/scratch/cw5167/workspace/tower_hanoi
cd "$REPO"
export OMP_NUM_THREADS=4
PY="$REPO/.venv/bin/python"
B="${BUNDLE:?set BUNDLE}"
OUT="$K/robot_rehearsal/session_${SLURM_JOB_ID}"
PORT=$((20000 + SLURM_JOB_ID % 20000))  # one port per job: two rehearsals on one node must not share a server
mkdir -p "$OUT"
echo "job $SLURM_JOB_ID on $(hostname -s), commit $(git rev-parse --short HEAD); server flags: ${SERVER_FLAGS:-}; runner flags: ${RUNNER_FLAGS:-}"
PYTHONPATH="$REPO" "$PY" -m app.vjepa_hanoi.goal_planner --fname "$B/config.yaml" --checkpoint "$B/weights.pt" \
  --goal_bundle "$B/goals.npz" --protocol ${PROTOCOL:-paper} ${SERVER_FLAGS:-} --host 127.0.0.1 --port $PORT > "$OUT/server.log" 2>&1 &
SERVER=$!
for i in $(seq 1 120); do kill -0 "$SERVER" 2>/dev/null || { echo "server exited"; tail -5 "$OUT/server.log"; exit 1; }
  curl -s http://127.0.0.1:$PORT/health > "$OUT/health.json" 2>/dev/null && break; sleep 5; done
echo "health: $(head -c 600 "$OUT/health.json")"
if [ "${CLIENT:-one_move}" = goal ]; then  # camera: the AAAA->CCCC recording, from its first row
  PYTHONPATH="$TH:$REPO" "$PY" -m app.vjepa_hanoi.arm_goal_client --dry-run --protocol ${PROTOCOL:-paper} \
    --replay /scratch/cw5167/datasets/hanoi_wm_roundtrip_20260925_171442_AAAA_to_CCCC.h5 --tower-hanoi "$TH" \
    --server http://127.0.0.1:$PORT --log "$OUT/runs/goal_run.npz" ${RUNNER_FLAGS:-} 2>&1 | grep -v Warning
else
  PYTHONPATH="$TH:$REPO" "$PY" -m app.vjepa_hanoi.one_move_test --dry-run --replay-archive "$K/data/expert6_first_move/test.npz" \
    --yes --tower-hanoi "$TH" --server http://127.0.0.1:$PORT --out "$OUT/runs" ${RUNNER_FLAGS:-} 2>&1 | grep -v Warning
fi
echo "runner exit ${PIPESTATUS[0]}"
kill "$SERVER"
ls -la "$OUT/runs" 2>/dev/null | head -20

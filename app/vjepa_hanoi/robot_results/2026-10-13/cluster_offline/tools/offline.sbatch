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
#                                                        rows in stages 3-4 only) or ROWS=all (whole move);
#                                                        MODEL=fps4 (default, A-firstmove) or MODEL=fps1 (the same run
#                                                        trained at data.fps 1; outputs get the prefix fps1_, rows
#                                                        every 8 raw rows as for fps4, each planning a 1 s step) or
#                                                        MODEL=jfps1 (run J: fps 1 from the JEPA-WMs corrected init, no
#                                                        state token; prefix jfps1_) or MODEL=efps1 (fps 1, official
#                                                        init, all expert moves; prefix efps1_) or MODEL=efps2 (the same at fps 2); MODEL may list several models,
#                                                        e.g. MODEL="jfps1 fps4", each run with the same SETTINGS;
#                                                        STRIDE=n evaluates rows n raw rows apart (default: one model
#                                                        step at fps 4, 8 rows for the fps-1 models)
#   sbatch --export=ALL,TASK=hover,MODEL=fps1 <this file> hover_probe only (for a model other than fps4)
#   sbatch --export=ALL,TASK=replay,MODEL=fps1,SETTINGS="base_h1 cat_stages_h1" <this file>
#                                                        descent_replay.py: closed loop from the hover on recorded frames
#                                                        (its own setting names); -> replay/<model>_replay.json
set -uo pipefail  # one failing setting does not stop the others
REPO=/scratch/cw5167/workspace/vjepa2
K=/scratch/cw5167/checkpoints/vjepa2_ac_hanoi
R=$REPO/app/vjepa_hanoi/robot_results/2026-10-13
OUT=$R/cluster_offline
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
PY="$REPO/.venv/bin/python"
model() {  # name -> CFG CKPT PFX DENSE
  case "$1" in
    fps4) CFG=configs/train/vitg16/hanoi-firstmove-ac-ft-256px-8f.yaml; CKPT=$K/A_firstmove_vitg_finetune_v1/best.pt
          PFX=""; DENSE="" ;;
    fps1) CFG=configs/train/vitg16/hanoi-firstmove-ac-ft-256px-8f-1fps.yaml
          CKPT=$K/A_firstmove_1fps_vitg_finetune_v1/best.pt; PFX=fps1_; DENSE="--row_stride 8" ;;
    jfps1) CFG=configs/train/vitg16/hanoi-firstmove-ac-ft-256px-8f-1fps-jepawms.yaml
           CKPT=$K/J_firstmove_1fps_vitg_finetune_v1/best.pt; PFX=jfps1_; DENSE="--row_stride 8" ;;
    efps1) CFG=configs/train/vitg16/hanoi-expert6all-ac-ft-256px-8f-1fps.yaml
           CKPT=$K/A_expert6all_1fps_vitg_finetune_v1/best.pt; PFX=efps1_; DENSE="--row_stride 8" ;;
    efps2) CFG=configs/train/vitg16/hanoi-expert6all-ac-ft-256px-8f-2fps.yaml
           CKPT=$K/A_expert6all_2fps_vitg_finetune_v1/best.pt; PFX=efps2_; DENSE="--row_stride 8" ;;
    *) echo "unknown MODEL $1" >&2; exit 2 ;;
  esac
}
MODELS="${MODEL:-fps4}"
stride() { if [ -n "${STRIDE:-}" ]; then DENSE="--row_stride ${STRIDE}"; fi; }
model "${MODELS%% *}"
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
    jw) echo "--protocol jepawms" ;;                       # JEPA-WMs planner (H3, 300x15, L2, ...), paper's 3 goals
    jw_stages) echo "--protocol jepawms --goals stages" ;;
    data_h1) echo "--sampling data" ;;                     # samples inside the training actions' range
    data_stages_h3) echo "--sampling data --goals stages --rollout 3" ;;
    jw_data_stages) echo "--protocol jepawms --sampling data --goals stages" ;;
    data_stages_h1) echo "--sampling data --goals stages" ;;
    data_stages_h2) echo "--sampling data --goals stages --rollout 2" ;;
    aid_data_stages_all_h2) echo "--sampling data --goals stages --hold_open --hold_xy --hold_grasp --rollout 2" ;;
    cat_h1) echo "--sampling cat" ;;                       # categorical over 51 training-action codes, paper3 goals
    cat_h2) echo "--sampling cat --rollout 2" ;;
    cat_stages_h1) echo "--sampling cat --goals stages" ;;   # H1, H2: every code sequence scored (exact)
    cat_stages_h2) echo "--sampling cat --goals stages --rollout 2" ;;
    cat_stages_h3) echo "--sampling cat --goals stages --rollout 3 --samples 400" ;;   # categorical CEM, prior ** 0.5
    cat_stages_h3_t0) echo "--sampling cat --goals stages --rollout 3 --samples 400 --cat_temp 0" ;;  # uniform start
    cat_stages_h3_t1) echo "--sampling cat --goals stages --rollout 3 --samples 400 --cat_temp 1" ;;  # data frequencies
    cat36_stages_h1) echo "--sampling cat --goals stages --cat_max_mm 36" ;;   # codes <= 36 mm (arm step rule at fps 1)
    cat36_stages_h2) echo "--sampling cat --goals stages --cat_max_mm 36 --rollout 2" ;;
    aid_cat_stages_h2) echo "--sampling cat --goals stages --hold_xy --hold_grasp --rollout 2" ;;
    aid_cat_stages_all_h2) echo "--sampling cat --goals stages --hold_open --hold_xy --hold_grasp --rollout 2" ;;
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
elif [ "${TASK}" = hover ]; then
  "$PY" "$OUT/tools/hover_probe.py" --fname "$CFG" --checkpoint "$CKPT" --out "$OUT/probes/${PFX}hover_probe.jsonl" \
    2>&1 | grep -v Warning | tee "$OUT/probes/${PFX}hover_probe.txt"
elif [ "${TASK}" = replay ]; then  # closed loop on recorded frames from the hover (descent_replay.py header)
  mkdir -p "$OUT/replay"
  for m in ${MODELS}; do
    model "$m"
    "$PY" "$OUT/tools/descent_replay.py" --fname "$CFG" --checkpoint "$CKPT" --settings "${SETTINGS// /,}" --max_steps 2 \
      --out "$OUT/replay/${m}_replay.json" 2>&1 | grep -v Warning
  done
elif [ "${TASK}" = ppe ]; then
  EVAL="--eval_stages 3,4"; TAG=descent
  if [ "${ROWS:-descent}" = all ]; then EVAL=""; TAG=all; fi
  for m in ${MODELS}; do
    model "$m"; stride
    for s in ${SETTINGS}; do
      flags=$(setting "$s")
      echo "=== $m $s: $flags $EVAL"
      "$PY" -m app.vjepa_hanoi.paper_protocol_eval --fname "$CFG" \
        --checkpoint "$CKPT" --archive $K/data/expert6_first_move/test.npz \
        --goal_archive $K/data/expert6_first_move/train.npz $EVAL $DENSE $flags --out "$OUT/ppe/${PFX}${TAG}_${s}.json" \
        2>&1 | grep -v Warning
    done
  done
fi

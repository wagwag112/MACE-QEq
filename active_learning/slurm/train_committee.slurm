#!/bin/bash
#SBATCH --job-name=train_committee
#SBATCH --partition=himsa40,requeue-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -eo pipefail


# ============================================================
# PATHS / ENVIRONMENT
# ============================================================

ROOT="/scratch/tmp/hdang/projects/${ANODE:?ANODE missing}_${SE:?SE missing}"
AL_DIR="/scratch/tmp/hdang/projects/active_learning"

source "${AL_DIR}/slurm/cluster_config.sh"
load_gpu_environment

cd "$AL_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1


# ============================================================
# CHECK EXECUTABLES
# ============================================================

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable missing:"
    echo "  $PYTHON"
    exit 1
fi

if [[ ! -x "$MACE_RUN_TRAIN" ]]; then
    echo "ERROR: mace_run_train executable missing:"
    echo "  $MACE_RUN_TRAIN"
    exit 1
fi


# ============================================================
# READ CONFIG
# ============================================================

eval "$(
"$PYTHON" - <<'PY'
import shlex
import al_config as cfg

committee_seeds = [
    seed
    for seed in cfg.COMMITTEE_SEEDS
    if seed != cfg.BASE_SEED
]

values = {
    "ROUND": cfg.ROUND,
    "TRAIN_SHORT": cfg.OUT_TRAIN_SHORT,
    "VALID_SHORT": cfg.OUT_VALID_SHORT,
    "COMMITTEE_DIR": cfg.COMMITTEE_NEXT_DIR,
    "SEED_LIST": " ".join(str(x) for x in committee_seeds),
}

for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"


# ============================================================
# ARRAY SEED
# ============================================================

read -r -a SEEDS <<< "$SEED_LIST"

TASK_ID="${SLURM_ARRAY_TASK_ID}"

if (( TASK_ID < 0 || TASK_ID >= ${#SEEDS[@]} )); then

    echo "ERROR: invalid committee array task:"
    echo "  ${TASK_ID}"

    exit 1
fi

SEED="${SEEDS[$TASK_ID]}"

NAME="MACE_short_v${ROUND}_seed${SEED}"

WORK_DIR="${COMMITTEE_DIR}/seed_${SEED}"


# ============================================================
# LOGGING
# ============================================================

LOG_DIR="${AL_DIR}/logs/v${ROUND}/committee"

mkdir -p "$LOG_DIR"

LOG_OUT="${LOG_DIR}/seed_${SEED}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out"
LOG_ERR="${LOG_DIR}/seed_${SEED}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.err"

exec > >(tee "$LOG_OUT") \
     2> >(tee "$LOG_ERR" >&2)


echo "======================================================================"
echo "MACE ACTIVE LEARNING COMMITTEE TRAINING"
echo "======================================================================"
echo "Round       : v${ROUND}"
echo "Seed        : ${SEED}"
echo "Model name  : ${NAME}"
echo "Job         : ${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
echo "Node        : ${SLURMD_NODENAME:-UNKNOWN}"
echo "Train file  : ${TRAIN_SHORT}"
echo "Valid file  : ${VALID_SHORT}"
echo "Output dir  : ${WORK_DIR}"
echo "Start       : $(date)"
echo "======================================================================"


# ============================================================
# INPUT CHECKS
# ============================================================

if [[ ! -s "$TRAIN_SHORT" ]]; then

    echo "ERROR: short-range training dataset missing:"
    echo "  $TRAIN_SHORT"

    exit 1
fi


if [[ ! -s "$VALID_SHORT" ]]; then

    echo "ERROR: short-range validation dataset missing:"
    echo "  $VALID_SHORT"

    exit 1
fi


# ============================================================
# PROTECT EXISTING COMMITTEE MEMBER
# ============================================================

RESTART_ARGS=()
FINAL_MODEL="${WORK_DIR}/checkpoints/${NAME}_run-${SEED}.model"

if [[ -e "$WORK_DIR" ]]; then
    if find "$WORK_DIR" -mindepth 1 -print -quit | grep -q .; then
        if [[ -s "$FINAL_MODEL" ]]; then
            echo "Final model already exists: $FINAL_MODEL"
            echo "Nothing to resume."
            exit 0
        fi
        shopt -s nullglob
        CHECKPOINTS=("$WORK_DIR"/checkpoints/"${NAME}_run-${SEED}_epoch-"*.pt)
        shopt -u nullglob
        if (( ${#CHECKPOINTS[@]} == 0 )); then
            echo "ERROR: output directory is non-empty but contains no compatible checkpoint: $WORK_DIR"
            exit 1
        fi
        echo "Compatible MACE checkpoint detected. Resuming existing generation with --restart_latest."
        RESTART_ARGS=(--restart_latest)
    fi
fi


mkdir -p \
    "$WORK_DIR/checkpoints" \
    "$WORK_DIR/logs" \
    "$WORK_DIR/results"

cd "$WORK_DIR"


# ============================================================
# GPU CHECK
# ============================================================

"$PYTHON" - <<'PY'
import torch

print("Torch version :", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("CUDA device   :", torch.cuda.get_device_name(0))
PY


# ============================================================
# TRAIN COMMITTEE MEMBER
#
# Independent training from scratch.
# Only the random seed differs between committee members.
#
# No --restart_latest.
# No SWA/stage-two.
# ============================================================

echo
echo "======================================================================"
echo "START COMMITTEE TRAINING"
echo "======================================================================"

"$MACE_RUN_TRAIN" \
    --name="$NAME" \
    --seed="$SEED" \
    --train_file="$TRAIN_SHORT" \
    --valid_file="$VALID_SHORT" \
    --energy_key="REF_energy" \
    --forces_key="REF_forces" \
    --model="MACE" \
    --r_max=6.0 \
    --hidden_irreps="128x0e + 128x1o" \
    --num_interactions=2 \
    --max_ell=3 \
    --correlation=3 \
    --batch_size=4 \
    --valid_batch_size=4 \
    --energy_weight=1.0 \
    --forces_weight=100.0 \
    --E0s="average" \
    --scaling="rms_forces_scaling" \
    --optimizer="adam" \
    --lr=0.01 \
    --weight_decay=5e-7 \
    --amsgrad \
    --scheduler="ReduceLROnPlateau" \
    --lr_factor=0.8 \
    --scheduler_patience=50 \
    --ema \
    --ema_decay=0.99 \
    --max_num_epochs=100 \
    --patience=2048 \
    --clip_grad=10.0 \
    --default_dtype="float32" \
    --device="cuda" \
    "${RESTART_ARGS[@]}"


# ============================================================
# FINAL MODEL CHECK
# ============================================================

MODEL_FILE="${WORK_DIR}/checkpoints/${NAME}_run-${SEED}.model"

if [[ ! -s "$MODEL_FILE" ]]; then

    echo
    echo "ERROR: expected committee model was not produced:"
    echo "  $MODEL_FILE"

    exit 1
fi


echo
echo "======================================================================"
echo "COMMITTEE TRAINING COMPLETED"
echo "======================================================================"
echo "Round : v${ROUND}"
echo "Seed  : ${SEED}"
echo "Model : ${MODEL_FILE}"
echo "End   : $(date)"
echo "======================================================================"
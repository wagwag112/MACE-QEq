#!/bin/bash
#SBATCH -A MST115294
#SBATCH -J constP1_4_0
#SBATCH -p 8gpus
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

set -eo pipefail

PROJ_ROOT="/work/hoang0000/Li_Li3N"
CLUSTER_ENV="/work/hoang0000/Li_LPSC/MACE_QEq/active_learning/cluster_config.sh"
[[ -f "${CLUSTER_ENV}" ]] || CLUSTER_ENV="${PROJ_ROOT}/active_learning/cluster_config.sh"
DRIVER="${PROJ_ROOT}/active_learning/stages/run_md.py"

TASK_NAME="constP1_6_-2_new"
MODE="constp"
STRUCT="${PROJ_ROOT}/structures/Li_Li3N_Li_new.vasp"
MODEL="${PROJ_ROOT}/trained_models/train_short_v1/MACE_short_v1.model"
STEPS=100000
RESTART=1

TEMP_K=300
TIMESTEP_FS=1.0
SAVE_EVERY=100
COLLISION_CHECK_EVERY=100
SEED=20261013
DEVICE="cuda"

PLANES_PER_SIDE=3
PLANE_TOL=0.35
PAIR_CAPACITY_FACTOR=1.15

ACTIVE_METAL="Li"
ANION_CUTOFFS='{"N": 2.50, "P": 2.80, "S": 2.90, "Cl": 3.05}'
RCUT_METAL=3.50

BOTTOM_EXTERNAL_CHI=6.0
UPPER_EXTERNAL_CHI=-2.0

LOG_DIR="${PROJ_ROOT}/logs"
OUT_DIR="${PROJ_ROOT}/md_validation/${TASK_NAME}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
LOG_OUT="${LOG_DIR}/md_${TASK_NAME}_${SLURM_JOB_ID}.out"
LOG_ERR="${LOG_DIR}/md_${TASK_NAME}_${SLURM_JOB_ID}.err"
exec > >(tee "${LOG_OUT}") 2> >(tee "${LOG_ERR}" >&2)

if [[ ! -f "${CLUSTER_ENV}" ]]; then
    echo "ERROR: Cluster config not found: ${CLUSTER_ENV}" >&2
    exit 1
fi

source "${CLUSTER_ENV}"
load_gpu_environment

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
PYTHON="${PYTHON:-${MACE_ENV}/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: Python executable missing or non-executable: ${PYTHON}" >&2
    exit 1
fi
if [[ ! -s "${STRUCT}" ]]; then
    echo "ERROR: Structure missing or empty: ${STRUCT}" >&2
    exit 1
fi
if [[ ! -s "${MODEL}" ]]; then
    echo "ERROR: Model checkpoint missing or empty: ${MODEL}" >&2
    exit 1
fi

EXPECTED_FRAMES=$((STEPS / SAVE_EVERY + 1))
if [[ "${RESTART}" != "1" && -s "${OUT_DIR}/md.extxyz" ]]; then
    DONE_FRAMES=$(grep -c "Lattice=" "${OUT_DIR}/md.extxyz" || true)
    if (( DONE_FRAMES >= EXPECTED_FRAMES - 2 )); then
        echo "[SKIP] ${TASK_NAME} already complete: ${DONE_FRAMES}/${EXPECTED_FRAMES} frames."
        exit 0
    fi
fi

if [[ "${RESTART}" == "0" && -s "${OUT_DIR}/md.traj" ]]; then
    echo "ERROR: Existing trajectory detected. Refusing fresh run."
    exit 1
fi

cd "${PROJ_ROOT}"

CMD_ARGS=(
    "--mode=${MODE}"
    "--structure=${STRUCT}"
    "--model=${MODEL}"
    "--temperature=${TEMP_K}"
    "--steps=${STEPS}"
    "--timestep-fs=${TIMESTEP_FS}"
    "--save-every=${SAVE_EVERY}"
    "--collision-check-every=${COLLISION_CHECK_EVERY}"
    "--seed=${SEED}"
    "--device=${DEVICE}"
    "--planes-per-side=${PLANES_PER_SIDE}"
    "--plane-tol=${PLANE_TOL}"
    "--pair-capacity-factor=${PAIR_CAPACITY_FACTOR}"
    "--active-metal=${ACTIVE_METAL}"
    "--anion-cutoffs=${ANION_CUTOFFS}"
    "--rcut-metal=${RCUT_METAL}"
    "--output=${OUT_DIR}"
)

if [[ "${RESTART}" == "1" ]]; then
    CMD_ARGS+=("--restart")
fi

if [[ "${MODE}" == "constp" ]]; then
    CMD_ARGS+=(
        "--bottom-external-chi=${BOTTOM_EXTERNAL_CHI}"
        "--upper-external-chi=${UPPER_EXTERNAL_CHI}"
    )
fi

echo "[MD] task=${TASK_NAME} mode=${MODE}"
echo "[MD] structure=${STRUCT}"
echo "[MD] model=${MODEL}"
echo "[MD] steps=${STEPS} timestep_fs=${TIMESTEP_FS} temperature_K=${TEMP_K}"
echo "[MD] output=${OUT_DIR}"

"${PYTHON}" -u "${DRIVER}" "${CMD_ARGS[@]}"


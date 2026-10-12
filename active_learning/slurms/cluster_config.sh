#!/bin/bash
#
# Cluster-specific software configuration for the MACE-QEq
# active-learning workflow.
#
# Scientific parameters do NOT belong here.
# Edit al_config.py for active-learning settings.
#

# ============================================================
# MACE / Python environment
# ============================================================

MINIFORGE_ACTIVATE="/scratch/tmp/hdang/software/miniforge3/bin/activate"
MACE_ENV="/scratch/tmp/hdang/software/envs/mace_qeq"

PYTHON="${MACE_ENV}/bin/python"
MACE_RUN_TRAIN="${MACE_ENV}/bin/mace_run_train"


# ============================================================
# GPU software stack
# ============================================================

load_gpu_environment() {

    module purge --force
    module load palma/2023b
    module load CUDA/12.4.0

    source "${MINIFORGE_ACTIVATE}"
    conda activate "${MACE_ENV}"

    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export PYTHONUNBUFFERED=1

    check_mace_environment
}


# ============================================================
# CPU software stack
# ============================================================

load_cpu_environment() {
    export JAX_PLATFORMS="cpu"

    module purge --force
    module load palma/2023b

    source "${MINIFORGE_ACTIVATE}"
    conda activate "${MACE_ENV}"

    export PYTHONUNBUFFERED=1

    check_mace_environment
}


# ============================================================
# Sanity checks
# ============================================================

check_mace_environment() {

    if [[ ! -x "${PYTHON}" ]]; then
        echo "ERROR: Python executable missing:"
        echo "  ${PYTHON}"
        return 1
    fi

    if [[ ! -x "${MACE_RUN_TRAIN}" ]]; then
        echo "ERROR: mace_run_train executable missing:"
        echo "  ${MACE_RUN_TRAIN}"
        return 1
    fi
}

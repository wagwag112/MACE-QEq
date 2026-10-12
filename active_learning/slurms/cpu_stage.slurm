#!/bin/bash
#SBATCH --job-name=al_cpu_stage
#SBATCH --partition=express,normal,long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

ROOT="/scratch/tmp/hdang/projects/Li_Li6PS5Cl"
AL_DIR="${ROOT}/active_learning"

cd "$AL_DIR"

if [[ -z "${AL_STAGE:-}" ]]; then
    echo "ERROR: AL_STAGE is not defined."
    exit 1
fi

case "$AL_STAGE" in
    candidate_selection)
        MODULE="stages.select_candidates"
        ;;
    collect_dft)
        MODULE="stages.collect_dft"
        ;;
    merge_dataset)
        MODULE="stages.merge_dataset"
        ;;
    *)
        echo "ERROR: unsupported AL_STAGE: $AL_STAGE"
        exit 1
        ;;
esac

source "${AL_DIR}/slurm/cluster_config.sh"
load_cpu_environment

ROUND=$("$PYTHON" - <<'PY'
import al_config as cfg
print(cfg.ROUND)
PY
)

LOG_DIR="${AL_DIR}/logs/v${ROUND}/${AL_STAGE}"
mkdir -p "$LOG_DIR"

LOG_OUT="${LOG_DIR}/${AL_STAGE}_${SLURM_JOB_ID}.out"
LOG_ERR="${LOG_DIR}/${AL_STAGE}_${SLURM_JOB_ID}.err"

exec > >(tee "$LOG_OUT") \
     2> >(tee "$LOG_ERR" >&2)

echo "======================================================================"
echo "ACTIVE LEARNING CPU STAGE"
echo "Round    : v${ROUND}"
echo "Stage    : ${AL_STAGE}"
echo "Module   : ${MODULE}"
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : ${SLURMD_NODENAME:-UNKNOWN}"
echo "Start    : $(date)"
echo "======================================================================"

MODULE_FILE="${AL_DIR}/${MODULE//.//}.py"

if [[ ! -s "$MODULE_FILE" ]]; then
    echo "ERROR: stage module missing or empty:"
    echo "$MODULE_FILE"
    exit 1
fi

"$PYTHON" -u -m "$MODULE"

echo
echo "======================================================================"
echo "CPU STAGE COMPLETED"
echo "Round : v${ROUND}"
echo "Stage : ${AL_STAGE}"
echo "End   : $(date)"
echo "======================================================================"
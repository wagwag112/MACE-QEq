#!/bin/bash
#SBATCH --job-name=al_qeq_residual
#SBATCH --partition=himsa40,requeue-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

ROOT="/scratch/tmp/hdang/projects/Li_Li6PS5Cl"
AL_DIR="${ROOT}/active_learning"

cd "$AL_DIR"

source "${AL_DIR}/slurm/cluster_config.sh"
load_gpu_environment

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

eval "$(
"$PYTHON" - <<'PY'
import shlex
import al_config as cfg

values = {
    "ROUND": cfg.ROUND,
    "DFT_FILE": cfg.DFT_FILE,
    "SHORT_FILE": cfg.SHORT_FILE,
    "QEQ_REPORT": cfg.QEQ_REPORT,
    "PROJECT_ROOT": cfg.PROJECT_ROOT,
}

for key, value in values.items():
    print(
        f"{key}="
        + shlex.quote(str(value))
    )
PY
)"

LOG_DIR="${AL_DIR}/logs/v${ROUND}/qeq_residual"
mkdir -p "$LOG_DIR"

LOG_OUT="${LOG_DIR}/qeq_${SLURM_JOB_ID}.out"
LOG_ERR="${LOG_DIR}/qeq_${SLURM_JOB_ID}.err"

exec > >(tee "$LOG_OUT") \
     2> >(tee "$LOG_ERR" >&2)

BUILDER="${AL_DIR}/stages/build_qeq_residual.py"

echo "======================================================================"
echo "ACTIVE LEARNING QEQ RESIDUAL CONSTRUCTION"
echo "Round       : v${ROUND}"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : ${SLURMD_NODENAME:-UNKNOWN}"
echo "DFT input   : ${DFT_FILE}"
echo "Short out   : ${SHORT_FILE}"
echo "Report      : ${QEQ_REPORT}"
echo "Builder     : ${BUILDER}"
echo "Start       : $(date)"
echo "======================================================================"

if [[ ! -s "$DFT_FILE" ]]; then
    echo "ERROR: DFT dataset missing or empty:"
    echo "$DFT_FILE"
    exit 1
fi

if [[ ! -s "$BUILDER" ]]; then
    echo "ERROR: QEq builder missing or empty:"
    echo "$BUILDER"
    exit 1
fi

rm -f "$SHORT_FILE" "$QEQ_REPORT"
"$PYTHON" -u "$BUILDER" \
    "$DFT_FILE" \
    -o "$SHORT_FILE" \
    --report "$QEQ_REPORT"

if [[ ! -s "$SHORT_FILE" ]]; then
    echo
    echo "ERROR: residual dataset was not created:"
    echo "$SHORT_FILE"
    exit 1
fi

if [[ ! -s "$QEQ_REPORT" ]]; then
    echo
    echo "ERROR: QEq report was not created:"
    echo "$QEQ_REPORT"
    exit 1
fi

echo
echo "======================================================================"
echo "QEQ RESIDUAL CONSTRUCTION COMPLETED"
echo "Output : ${SHORT_FILE}"
echo "Report : ${QEQ_REPORT}"
echo "End    : $(date)"
echo "======================================================================"

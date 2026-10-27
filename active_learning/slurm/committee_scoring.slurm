#!/bin/bash
#SBATCH --job-name=score
#SBATCH --partition=himsa40,requeue-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -eo pipefail

ROOT="/scratch/tmp/hdang/projects/${ANODE:?ANODE missing}_${SE:?SE missing}"
AL_DIR="/scratch/tmp/hdang/projects/active_learning"

source "${AL_DIR}/slurm/cluster_config.sh"
load_gpu_environment

cd "$AL_DIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable not found or not executable:"
    echo "  $PYTHON"
    exit 1
fi

ROUND=$(
    "$PYTHON" - <<'PY'
import al_config
print(al_config.ROUND)
PY
)

ROOT="/scratch/tmp/hdang/projects/${ANODE:?ANODE missing}_${SE:?SE missing}"
LOG_DIR="${AL_DIR}/logs/v${ROUND}/committee_scoring"
mkdir -p "$LOG_DIR"

LOG_OUT="$LOG_DIR/score_${SLURM_JOB_ID}.out"
LOG_ERR="$LOG_DIR/score_${SLURM_JOB_ID}.err"

exec > >(tee "$LOG_OUT") \
     2> >(tee "$LOG_ERR" >&2)

echo "======================================================================"
echo "ACTIVE LEARNING COMMITTEE SCORING"
echo "Round      : v${ROUND}"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : ${SLURMD_NODENAME:-UNKNOWN}"
echo "Python     : ${PYTHON}"
echo "Start      : $(date)"
echo "======================================================================"

"$PYTHON" - <<'PY'
import sys
import torch
import ase

print("Python :", sys.executable)
print("Torch  :", torch.__version__)
print("CUDA   :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))
PY

echo
echo "======================================================================"
echo "RUNNING COMMITTEE SCORING"
echo "======================================================================"

"$PYTHON" -u -m stages.score_committee

echo
echo "======================================================================"
echo "COMMITTEE SCORING COMPLETED SUCCESSFULLY"
echo "End        : $(date)"
echo "======================================================================"
#!/bin/bash
#SBATCH --job-name=al_probe_score
#SBATCH --partition=gpuexpress,himsa40,requeue-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

ROOT="/scratch/tmp/hdang/projects/Li_Li6PS5Cl"
AL_DIR="${ROOT}/active_learning"

cd "$AL_DIR"

source "${AL_DIR}/slurm/cluster_config.sh"
load_gpu_environment

export PYTHONUNBUFFERED=1

ROUND=$(
"$PYTHON" - <<'PY'
import al_config as cfg
print(cfg.ROUND)
PY
)

LOG_DIR="${AL_DIR}/logs/v${ROUND}/probe_scoring"
mkdir -p "$LOG_DIR"

LOG_OUT="${LOG_DIR}/probe_scoring_${SLURM_JOB_ID}.out"
LOG_ERR="${LOG_DIR}/probe_scoring_${SLURM_JOB_ID}.err"

exec > >(tee "$LOG_OUT") \
     2> >(tee "$LOG_ERR" >&2)

echo "======================================================================"
echo "FRESH PROBE COMMITTEE SCORING"
echo "Round : v${ROUND}"
echo "Job   : ${SLURM_JOB_ID}"
echo "Node  : ${SLURMD_NODENAME:-UNKNOWN}"
echo "Start : $(date)"
echo "======================================================================"

"$PYTHON" -u -m stages.score_fresh_probe

echo
echo "======================================================================"
echo "PROBE SCORING COMPLETED"
echo "End : $(date)"
echo "======================================================================"

#!/bin/bash

#SBATCH --job-name=evaluate
#SBATCH --partition=gpuexpress,himsa40,requeue-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

ROOT="/scratch/tmp/hdang/projects/${ANODE:?ANODE missing}_${SE:?SE missing}"
AL_DIR="/scratch/tmp/hdang/projects/active_learning"

cd "$AL_DIR"

source "${AL_DIR}/slurm/cluster_config.sh"
load_gpu_environment

ROUND=$(
"$PYTHON" - <<'PY'
import al_config as cfg
print(cfg.ROUND)
PY
)

LOG_DIR="${AL_DIR}/logs/v${ROUND}/evaluation"
mkdir -p "$LOG_DIR"

LOG_OUT="${LOG_DIR}/evaluation_${SLURM_JOB_ID}.out"
LOG_ERR="${LOG_DIR}/evaluation_${SLURM_JOB_ID}.err"

exec > >(tee "$LOG_OUT") \
     2> >(tee "$LOG_ERR" >&2)

echo "======================================================================"
echo "ACTIVE LEARNING POST-TRAINING EVALUATION"
echo "Round : v${ROUND}"
echo "Job   : ${SLURM_JOB_ID}"
echo "Node  : ${SLURMD_NODENAME:-UNKNOWN}"
echo "Start : $(date)"
echo "======================================================================"

"$PYTHON" -u -m stages.evaluate_models

echo
echo "======================================================================"
echo "EVALUATION COMPLETED"
echo "End : $(date)"
echo "======================================================================"

#!/bin/bash
#SBATCH --job-name=fresh_probe
#SBATCH --partition=himsa40,requeue-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

ROOT="/scratch/tmp/hdang/projects/${ANODE:?ANODE missing}_${SE:?SE missing}"
AL_DIR="/scratch/tmp/hdang/projects/active_learning"

cd "$AL_DIR"

source "${AL_DIR}/slurm/cluster_config.sh"
load_gpu_environment

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

ROUND=$(
"$PYTHON" - <<'PY'
import al_config as cfg
print(cfg.ROUND)
PY
)

LOG_DIR="${AL_DIR}/logs/v${ROUND}/fresh_probe"
mkdir -p "$LOG_DIR"

LOG_OUT="${LOG_DIR}/fresh_probe_${SLURM_JOB_ID}.out"
LOG_ERR="${LOG_DIR}/fresh_probe_${SLURM_JOB_ID}.err"

exec > >(tee "$LOG_OUT") \
     2> >(tee "$LOG_ERR" >&2)

echo "======================================================================"
echo "ACTIVE LEARNING FRESH CONVERGENCE PROBE"
echo "Round : v${ROUND}"
echo "Job   : ${SLURM_JOB_ID}"
echo "Node  : ${SLURMD_NODENAME:-UNKNOWN}"
echo "Start : $(date)"
echo "======================================================================"

"$PYTHON" -u -m stages.run_fresh_probe

echo
echo "======================================================================"
echo "FRESH PROBE COMPLETED"
echo "End : $(date)"
echo "======================================================================"

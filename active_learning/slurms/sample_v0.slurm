#!/bin/bash
#SBATCH --job-name=al_v0_sample
#SBATCH --partition=himsa40,requeue-gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

ROOT="/scratch/tmp/hdang/projects/Li_Li6PS5Cl"
AL_DIR="${ROOT}/active_learning"

cd "$AL_DIR"

source "${AL_DIR}/slurm/cluster_config.sh"
load_gpu_environment

export OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

LOG_DIR="${AL_DIR}/logs/v0/sample_v0"
mkdir -p "$LOG_DIR"

LOG_OUT="${LOG_DIR}/sample_v0_${SLURM_JOB_ID}.out"
LOG_ERR="${LOG_DIR}/sample_v0_${SLURM_JOB_ID}.err"

exec > >(tee "$LOG_OUT") \
     2> >(tee "$LOG_ERR" >&2)

echo "======================================================================"
echo "ACTIVE LEARNING V0 MULTI-GPU STREAMING BOOTSTRAP"
echo "Allocated GPUs : $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Node           : ${SLURMD_NODENAME:-UNKNOWN}"
echo "Start          : $(date)"
echo "======================================================================"

"$PYTHON" -u "${AL_DIR}/stages/sample_v0.py"

echo
echo "======================================================================"
echo "V0 BOOTSTRAP COMPLETED"
echo "End   : $(date)"
echo "======================================================================"

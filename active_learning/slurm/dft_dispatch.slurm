#!/bin/bash
#SBATCH --job-name=dft_mgr
#SBATCH --partition=normal,long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

ROOT="/scratch/tmp/hdang/projects/${ANODE:?ANODE missing}_${SE:?SE missing}"
AL_DIR="/scratch/tmp/hdang/projects/active_learning"

source "${AL_DIR}/slurm/cluster_config.sh"
load_cpu_environment
cd "${AL_DIR}"

ROUND=$("$PYTHON" - <<'PY'
import al_config as cfg
print(cfg.ROUND)
PY
)

LOG_DIR="${AL_DIR}/logs/v${ROUND}/dft"
mkdir -p "$LOG_DIR"

LOG_OUT="${LOG_DIR}/dft_mgr_${SLURM_JOB_ID}.out"
LOG_ERR="${LOG_DIR}/dft_mgr_${SLURM_JOB_ID}.err"

exec > >(tee "$LOG_OUT") \
     2> >(tee "$LOG_ERR" >&2)

echo "======================================================================"
echo "ACTIVE LEARNING DFT DISPATCH MANAGER"
echo "Round    : v${ROUND}"
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : ${SLURMD_NODENAME:-UNKNOWN}"
echo "Start    : $(date)"
echo "======================================================================"

"$PYTHON" -u "${AL_DIR}/stages/dispatch_dft.py"

echo
echo "======================================================================"
echo "DFT DISPATCH MANAGER COMPLETED"
echo "End      : $(date)"
echo "======================================================================"

#!/bin/bash
#SBATCH --job-name=sample_bulk
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=3-8
#SBATCH --output=/scratch/tmp/hdang/projects/active_learning/slurm/logs/sample_bulk_%A_%a.out
#SBATCH --error=/scratch/tmp/hdang/projects/active_learning/slurm/logs/sample_bulk_%A_%a.err

set -euo pipefail

AL_DIR="/scratch/tmp/hdang/projects/active_learning"
source "$AL_DIR/slurm/cluster_config.sh"
load_cpu_environment

cd "$AL_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$PYTHON" -u stages/sample_v0.py

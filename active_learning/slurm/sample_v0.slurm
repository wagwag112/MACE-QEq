#!/bin/bash
#SBATCH --job-name=sample_v0
#SBATCH --partition=himsa40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/tmp/hdang/projects/active_learning/slurm/logs/sample_v0_%A_%a.out
#SBATCH --error=/scratch/tmp/hdang/projects/active_learning/slurm/logs/sample_v0_%A_%a.err

set -euo pipefail

AL_DIR="/scratch/tmp/hdang/projects/active_learning"
source "$AL_DIR/slurm/cluster_config.sh"
load_gpu_environment

cd "$AL_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$PYTHON" -u stages/sample_v0.py

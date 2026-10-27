#!/bin/bash
#SBATCH --job-name=sample_v0_fin
#SBATCH --partition=express,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/tmp/hdang/projects/active_learning/slurm/logs/sample_v0_fin_%j.out
#SBATCH --error=/scratch/tmp/hdang/projects/active_learning/slurm/logs/sample_v0_fin_%j.err

set -euo pipefail

AL_DIR="/scratch/tmp/hdang/projects/active_learning"

source "$AL_DIR/slurm/cluster_config.sh"
load_cpu_environment

cd "$AL_DIR"
"$PYTHON" -u stages/sample_v0.py --finalize

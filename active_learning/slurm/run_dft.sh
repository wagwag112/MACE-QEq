#!/bin/bash
set -uo pipefail

VASP_BIN="/scratch/tmp/hdang/software/vasp.6.4.3/bin/vasp_std"

# Suppress all Lmod output to STDERR to prevent slurm.err contamination
module --force purge >/dev/null 2>&1 || true
module load palma/2021b intel-compilers/2021.4.0 impi/2021.4.0 imkl/2021.4.0 >/dev/null 2>&1 || true

WORKDIR="${VASP_WORKDIR:-$(pwd)}"
cd "${WORKDIR}" || exit 1

export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ulimit -s unlimited

# Force libfabric (OFI) to use pure shared memory for single-node DFT
# Eliminate Infiniband/verbs network conflicts that can cause deadlocks on GPU/heterogeneous nodes
export FI_PROVIDER="shm"
export I_MPI_FABRICS="shm"
export I_MPI_SHM=1
export I_MPI_FALLBACK=0
export I_MPI_HYDRA_BOOTSTRAP="slurm"

# Use exactly 24 CPU cores
srun --mpi=pmi2 -n "${SLURM_NTASKS:-24}" "${VASP_BIN}" 2>/dev/null || \
srun -n "${SLURM_NTASKS:-24}" "${VASP_BIN}"
STATUS=$?

exit "${STATUS}"

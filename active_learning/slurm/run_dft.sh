#!/bin/bash
set -uo pipefail

VASP_BIN="/scratch/tmp/hdang/software/vasp.6.4.3/bin/vasp_std"

# Suppress all Lmod output to STDERR to prevent slurm.err contamination
module --force purge >/dev/null 2>&1 || true
module load palma/2021b intel-compilers/2021.4.0 impi/2021.4.0 imkl/2021.4.0 >/dev/null 2>&1 || true

WORKDIR="${VASP_WORKDIR:-$(pwd)}"
cd "${WORKDIR}" || exit 1

vasprun_valid() {
    [[ -s vasprun.xml ]] && tail -c 500 vasprun.xml 2>/dev/null | grep -q "</modeling>"
}

# Idempotent guard: never relaunch VASP over a completed vasprun.xml
if vasprun_valid; then
    echo "[IDEMPOTENT SKIP] valid vasprun.xml found; not rerunning VASP."
    exit 0
fi

export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ulimit -s unlimited

# Force libfabric (OFI) to use pure shared memory for single-node DFT
export FI_PROVIDER="shm"
export I_MPI_FABRICS="shm"
export I_MPI_SHM=1
export I_MPI_FALLBACK=0
export I_MPI_HYDRA_BOOTSTRAP="slurm"

# Use exactly 24 CPU cores
srun --mpi=pmi2 -n "${SLURM_NTASKS:-24}" "${VASP_BIN}" 2>/dev/null
STATUS=$?

# Fallback launcher only if the first attempt failed AND produced no valid result
if [[ $STATUS -ne 0 ]] && ! vasprun_valid; then
    srun -n "${SLURM_NTASKS:-24}" "${VASP_BIN}"
    STATUS=$?
fi

exit "${STATUS}"

#!/bin/bash -l

#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --job-name="manual-nsys"
#SBATCH --output=out/manual-nsys.%j.o
#SBATCH --error=out/manual-nsys.%j.e
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1

set -euo pipefail

PROFILE_DIR="${SLURM_SUBMIT_DIR}/profiles"

mkdir -p "${SLURM_SUBMIT_DIR}/out"
mkdir -p "${PROFILE_DIR}"

export PROFILE_DIR

export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_CUDAAWARE_MPI=1
export JULIA_CUDA_USE_COMPAT=false

# Allow Nsight to trigger from normal, unregistered NVTX strings.
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0

srun \
    --cpu-bind=cores \
    --export=ALL,USE_GPU=true \
    --uenv=julia/25.5:v1 \
    --view=juliaup,modules \
    bash -lc '
        module load cuda

        echo "rank=${SLURM_PROCID} host=$(hostname) cwd=$(pwd)"
        echo "profile output: ${PROFILE_DIR}"

        exec nsys profile \
            --trace=cuda,nvtx,mpi,osrt \
            --mpi-impl=mpich \
            --sample=none \
            --osrt-threshold=10000 \
            --capture-range=nvtx \
            --nvtx-capture="profile@*" \
            --capture-range-end=stop \
            --force-overwrite=true \
            --output="${PROFILE_DIR}/manual_${SLURM_JOB_ID}_rank%q{SLURM_PROCID}_host%h_pid%p" \
            julia --project="${SLURM_SUBMIT_DIR}" \
                  "${SLURM_SUBMIT_DIR}/manual.jl" \
                  --benchmark
    '
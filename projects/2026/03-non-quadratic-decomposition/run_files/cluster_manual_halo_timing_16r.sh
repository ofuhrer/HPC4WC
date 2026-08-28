#!/bin/bash -l
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --job-name="halo_timing_16"
#SBATCH --output=out/halo_timing_16.%j.o
#SBATCH --error=out/halo_timing_16.%j.e
#SBATCH --time=00:30:00
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --uenv=julia/25.5:v1
#SBATCH --gpus-per-task=1

export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_CUDAAWARE_MPI=1
export JULIA_CUDA_USE_COMPAT=false

topologies=(
    "16x1"
    "8x2"
    "4x4"
    "2x8"
    "1x16"
)

for topo in "${topologies[@]}"; do
    echo "=========================================="
    echo " Start srun with: --topology ${topo}"
    echo "=========================================="

    csv_file="output/halo_exchange_16_ranks.csv"

    srun --export=ALL,USE_GPU=true julia --project manual_time_halo.jl \
        --topology "${topo}" \
        --nx 32770 \
        --ny 8194 \
        --warmup 20 \
        --benchmark \
        --benchdir "${csv_file}"

    echo "Topologie ${topo} finished."
done
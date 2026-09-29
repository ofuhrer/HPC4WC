#!/bin/bash -l
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --job-name="halo_timing"
#SBATCH --output=out/halo_timing.%j.o
#SBATCH --error=out/halo_timing.%j.e
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --uenv=julia/25.5:v1
#SBATCH --gpus-per-task=1

export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_CUDAAWARE_MPI=1
export JULIA_CUDA_USE_COMPAT=false

topologies=(
    "4x1"
    "2x2"
    "1x4"
)

for topo in "${topologies[@]}"; do
    echo "=========================================="
    echo " Start srun with: --topology ${topo}"
    echo "=========================================="

    csv_file="output/halo_exchange_4_ranks.csv"

    srun --export=ALL,USE_GPU=true julia --project manual_time_halo.jl \
        --topology "${topo}" \
        --nx 16386 \
        --ny 4098 \
        --warmup 20 \
        --benchmark \
        --benchdir "${csv_file}"

    echo "Topologie ${topo} finished."
done
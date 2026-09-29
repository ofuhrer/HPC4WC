#!/bin/bash -l
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --job-name="manual"
#SBATCH --output=out/manual.%j.o
#SBATCH --error=out/manual.%j.e
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
    "(4,1)"
    "(2,2)"
    "(1,4)"
)

for topo in "${topologies[@]}"; do
    echo "=========================================="
    echo " Start srun with: --topo ${topo}"
    echo "=========================================="
    
    topo_clean=$(echo "${topo}" | tr -d '(),')
    csv_file="output/walltimes_4_ranks.csv"

    srun --export=ALL,USE_GPU=true julia --project manual.jl \
        --topo "${topo}" \
        --nx 16386 \
        --ny 4098 \
        --nt 200 \
        --warmup 20 \
        --runs 20 \
        --benchmark \
        --benchdir "${csv_file}"

    echo "Topology ${topo} finished."
done
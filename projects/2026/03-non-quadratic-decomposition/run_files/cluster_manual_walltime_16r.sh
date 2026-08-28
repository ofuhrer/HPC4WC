#!/bin/bash -l
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --job-name="manual"
#SBATCH --output=out/manual.%j.o
#SBATCH --error=out/manual.%j.e
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --uenv=julia/25.5:v1
#SBATCH --gpus-per-task=1

export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_CUDAAWARE_MPI=1 # IGG
export JULIA_CUDA_USE_COMPAT=false # IGG

# Define topologies in the format expected by manual.jl
topologies=(
    "(16,1)"
    "(8,2)"
    "(4,4)"
    "(2,8)"
    "(1,16)"
)


for topo in "${topologies[@]}"; do
    echo "=========================================="
    echo " Start srun with: --topo ${topo}"
    echo "=========================================="
    
    topo_clean=$(echo "${topo}" | tr -d '(),')
    csv_file="output/walltimes_16_ranks.csv"

    srun --export=ALL,USE_GPU=true julia --project manual.jl \
        --topo "${topo}" \
        --nx 32770 \
        --ny 8194 \
        --runs 15 \
        --benchmark \
        --benchdir "${csv_file}"

    echo "Topology ${topo} finished."
done


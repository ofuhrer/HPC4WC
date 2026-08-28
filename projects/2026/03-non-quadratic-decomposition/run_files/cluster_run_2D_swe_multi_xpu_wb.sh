#!/bin/bash -l
#SBATCH --account=julia-gpu-course2026-ethz
#SBATCH --job-name="swe_multi_xpu"
#SBATCH --output=out/swe__multi_xpu.%j.o
#SBATCH --error=out/swe_multi_xpu.%j.e
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1

export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_CUDAAWARE_MPI=1 # IGG
export JULIA_CUDA_USE_COMPAT=false # IGG

srun --uenv julia/25.5:v1 --view=juliaup julia --project src/baseline.jl

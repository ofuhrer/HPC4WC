#!/bin/bash -l
#SBATCH --job-name="stencil2d_cuda"
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun ./stencil2d_cuda.x --nx 128 --ny 128 --nz 64 --niters 1024

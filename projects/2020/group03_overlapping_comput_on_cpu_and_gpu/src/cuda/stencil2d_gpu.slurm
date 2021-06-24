#!/bin/bash -l
#SBATCH --job-name="stencil2d_gpu"
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

nx=1024
ny=1024
nz=64
niters=1024

for nxy in 128 256 512 1024
do
    srun ./stencil2d_gpu.x --nx $nxy --ny $nxy --nz $nz --niters $niters
done

#srun               ./stencil2d_gpu.x --nx $nx --ny $ny --nz $nz --niters $niters
#srun nvprof        ./stencil2d_gpu.x --nx $nx --ny $ny --nz $nz --niters $niters
#srun cuda-memcheck ./stencil2d_gpu.x --nx $nx --ny $ny --nz $nz --niters $niters


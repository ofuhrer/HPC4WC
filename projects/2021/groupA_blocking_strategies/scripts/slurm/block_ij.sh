#!/bin/bash -l
#SBATCH --job-name="job_name"
#SBATCH --account="class03"
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --output=../output/block_ij.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ../../build/src/executables
srun ./block_ij
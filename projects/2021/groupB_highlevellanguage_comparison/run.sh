#!/bin/bash -l
#SBATCH --job-name="benchmark_libraries"
#SBATCH --account="class03"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=spiasko@student.ethz.ch
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

srun python src/main.py --out data/jax.csv --libs jax --ns 8 16 32 64 128 256 512 1024
#srun python -m bohrium src/main.py --out data/bohrium.csv --libs numpy --ns 8 16 32 64 128 256 512 1024
#srun python src/main.py --out data/gt4py.csv --libs gt4py --ns 8 16 32 64 128 256 512 1024
#srun python src/main.py --out data/numpy.csv --libs numpy --ns 8 16 32 64 128 256 512 1024 

#!/bin/bash -l
#SBATCH --job-name="test"
#SBATCH --time=01:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --output=CPU.out
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

source /users/course22/HPC4WC_venv/bin/activate

export DEVITO_LANGUAGE=openmp
#1 2 3 4 5 6 7 8 9 10 11 12 
for i in 12
do
export OMP_NUM_THREADS=$i
srun python ../benchmark/benchmark_CPU.py
done



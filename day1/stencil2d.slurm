#!/bin/bash -l
#SBATCH --job-name="stencil2d.$USER"
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

exe=stencil2d.x
nx=128
ny=128
nz=64
num_iter=1024

echo "==== START OF EXECUTION `date +%s` ===="
srun ./${exe}+orig -nx ${nx} -ny ${ny} -nz ${nz} -num_iter ${num_iter}
echo "===== END OF EXECUTION `date +%s` ====="

echo "==== START OF PROFILING `date +%s` ===="
srun ./${exe} -nx ${nx} -ny ${ny} -nz ${nz} -num_iter ${num_iter}
echo "===== END OF PROFILING `date +%s` ====="

exit 0

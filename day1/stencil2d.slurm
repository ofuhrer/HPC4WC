#!/bin/bash -l
#SBATCH --job-name="stencil2d.$USER"
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --view=default

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ref_exe=stencil2d-orig.x
exe=stencil2d.x
nx=128
ny=128
nz=64
num_iter=1024

echo "==== START OF EXECUTION `date +%s` ===="
srun ./${ref_exe} --nx ${nx} --ny ${ny} --nz ${nz} --num_iter ${num_iter}
echo "===== END OF EXECUTION `date +%s` ====="

echo "==== START OF PROFILING `date +%s` ===="
srun ./${exe} --nx ${nx} --ny ${ny} --nz ${nz} --num_iter ${num_iter}
echo "===== END OF PROFILING `date +%s` ====="

exit 0

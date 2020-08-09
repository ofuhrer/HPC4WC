#!/bin/bash -l
#SBATCH --job-name        benchmark
#SBATCH --output          %x.out
#SBATCH --error           %x.err
#SBATCH --open-mode       truncate
#SBATCH --nodes           4
#SBATCH --constraint      gpu
#SBATCH --time            01:00:00
#SBATCH --partition       normal
#SBATCH --ntasks-per-core 1
#SBATCH --hint            nomultithread
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task   12

module load cce
module load gcc
module load intel
module load pgi
module load cudatoolkit

export GREASY_LOGFILE=${SLURM_JOB_NAME}.log
export GREASY_NWORKER_PER_NODE=${SLURM_NTASKS_PER_NODE}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_TARGET_OFFLOAD="MANDATORY"
export CRAY_CUDA_MPS=1

# see wrapper.sh for runtime parameters
greasy scripts/greasy.txt

#!/bin/bash -l
#SBATCH --job-name        data1
#SBATCH --output          %x.out
#SBATCH --error           %x.err
#SBATCH --open-mode       truncate
#SBATCH --nodes            2
#SBATCH --constraint      gpu
#SBATCH --time            00:30:00
#SBATCH --partition       normal
#SBATCH --hint            nomultithread
#SBATCH --ntasks-per-core  1
#SBATCH --ntasks-per-node  1
#SBATCH --cpus-per-task   12

prg_env=$(module list 2>&1 | sed -n -E -e 's/.*(PrgEnv-.*)/\1/p')
module switch ${prg_env} PrgEnv-cray
module load cudatoolkit
module load GREASY

export GREASY_LOGFILE=${SLURM_JOB_NAME}.log
export GREASY_NWORKER_PER_NODE=${SLURM_NTASKS_PER_NODE}
export OMP_NUM_THREAD=${SLURM_CPUS_PER_TASK}
export OMP_TARGET_OFFLOAD="MANDATORY"
export CRAY_CUDA_MPS=1

greasy scripts/data1.txt

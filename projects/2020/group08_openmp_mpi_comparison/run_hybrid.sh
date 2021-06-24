#!/bin/bash -l
#
#SBATCH --time=00:30:00
#SBATCH --ntasks=24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --ntasks-per-core=2

export OMP_NUM_THREADS=1
#Add opneMP tuning here
#export OMP_SCHEDULE="DYNAMIC"
echo "Hybrid" 

#srun ./stencil2d-mpiomp.x --nx 64 --ny 64 --nz 32 --num_iter 128
srun ./stencil2d-mpiomp.x --nx 128 --ny 128 --nz 64 --num_iter 1024

#NOTES
#MPI ranks = ntasks
#divide mpi ranks over nodes = ntasks-per-node
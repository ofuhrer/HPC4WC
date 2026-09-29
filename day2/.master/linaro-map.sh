#!/bin/bash

export OMP_NUM_THREADS=72
export OMP_PLACES=cores
export OMP_PROC_BIND=close

#LD_LIBRARY_PATH=/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/cray-gtl-8.1.30-yjzx5mja37woqidkb7lj6lpbu3omqrsr/lib:$LD_LIBRARY_PATH /capstor/scratch/cscs/piccinal/linaro/25.0/bin/map -n 1 --mpi=slurm --mpiargs="-N 1 --hint=nomultithread --distribution=block:block --time=0:10:00" --profile $@

LD_LIBRARY_PATH=/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/cray-gtl-8.1.30-yjzx5mja37woqidkb7lj6lpbu3omqrsr/lib:$LD_LIBRARY_PATH /capstor/scratch/cscs/piccinal/linaro/25.0/bin/map -n 1 --mpi=slurm --mpiargs="-N 1 --time=0:10:00" --profile $@

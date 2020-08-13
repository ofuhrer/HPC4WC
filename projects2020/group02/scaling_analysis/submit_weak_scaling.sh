#!/bin/bash -l
#
#SBATCH --job-name="cubed_sphere_weak_scaling"
#SBATCH --time=00:25:00
#SBATCH --nodes=18
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --output=cubed_sphere_weak_scaling.%j.out
#SBATCH --error=cubed_sphere_weak_scaling.%j.err
  
echo "---------------------------------------------------------------------------------------------"
echo "Weak scaling runs"
echo "---------------------------------------------------------------------------------------------"
echo ""
for size in 60 120
do
    for multi in 1 2 3 4 5 6
    do
        totalsize=$(($size * $multi))
        ntasks=$((6 * $multi * $multi)) 
        srun --unbuffered -n$ntasks python3 ./stencil3d-mpi.py --nx=$totalsize --ny=$totalsize --nz=60 --num_iter=1020
        srun --unbuffered -n$ntasks python3 ./stencil3d-mpi.py --nx=$totalsize --ny=$totalsize --nz=60 --num_iter=1020
        srun --unbuffered -n$ntasks python3 ./stencil3d-mpi.py --nx=$totalsize --ny=$totalsize --nz=60 --num_iter=1020
        echo ""
    done
    echo "---------------------------------------------------------------------------------------------"
    echo ""
done

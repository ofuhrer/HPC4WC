#!/bin/bash -l
#
#SBATCH --job-name="cubed_sphere_strong_scaling"
#SBATCH --time=00:15:00
#SBATCH --nodes=18
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --output=cubed_sphere_strong_scaling.%j.out
#SBATCH --error=cubed_sphere_strong_scaling.%j.err

echo "---------------------------------------------------------------------------------------------"
echo "Strong scaling runs"
echo "---------------------------------------------------------------------------------------------"
echo ""
for size in 120 240
do
    for ntasks in 6 24 54 96 150 216
    do
        srun --unbuffered -n$ntasks python3 ./stencil3d-mpi.py --nx=$size --ny=$size --nz=60 --num_iter=1020
        srun --unbuffered -n$ntasks python3 ./stencil3d-mpi.py --nx=$size --ny=$size --nz=60 --num_iter=1020
        srun --unbuffered -n$ntasks python3 ./stencil3d-mpi.py --nx=$size --ny=$size --nz=60 --num_iter=1020
        echo ""
    done
    echo "---------------------------------------------------------------------------------------------"
    echo ""
done


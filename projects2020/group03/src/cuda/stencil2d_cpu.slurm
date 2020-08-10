#!/bin/bash -l
#SBATCH --job-name="stencil2d_cpu"
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

nx=128
ny=128
nz=64
niter=1024

for nxy in 128 256 512 1024
do
    for z_slices_on_cpu in 64 32 1
    do
        srun ./stencil2d_cpu.x --nx $nxy --ny $nxy --nz $nz --num_iter $niter --z_slices_on_cpu $z_slices_on_cpu
    done
done

#nthreads=1
#echo "out = [None] * 21" > out_cpu.txt
#for nthreads in `seq 6 12` ; do
#export OMP_NUM_THREADS=$nthreads
#  ncores=$nthreads
#  if [ $nthreads -gt 12 ] ; then
#    ncores=12
#  fi
#  srun -n 1 -c $ncores ./stencil2d_cpu.x --nx $nx --ny $ny --nz $nz --num_iter $niter | sed "s/data =/out[$nthreads] =/g" >> out_cpu.txt
#done

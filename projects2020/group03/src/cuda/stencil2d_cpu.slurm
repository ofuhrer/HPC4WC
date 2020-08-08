#!/bin/bash -l
#SBATCH --job-name="stencil2d"
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

nx=128
ny=128
nz=64
niter=1024

srun ./stencil2d_cpu.x --nx $nx --ny $ny --nz $nz --num_iter $niter >> report_cpu.txt

nthreads=1
echo "out = [None] * 21" > out_cpu.txt
for nthreads in `seq 6 12` ; do
export OMP_NUM_THREADS=$nthreads
  ncores=$nthreads
  if [ $nthreads -gt 12 ] ; then
    ncores=12
  fi
  srun -n 1 -c $ncores ./stencil2d_cpu.x --nx $nx --ny $ny --nz $nz --num_iter $niter | sed "s/data =/out[$nthreads] =/g" >> out_cpu.txt
done

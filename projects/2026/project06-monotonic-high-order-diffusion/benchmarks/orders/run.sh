#!/bin/bash
#
# Runtime against filter order, limiter on and off.
#
# Authors: Stefanie Boersig <stefanie.boersig@env.ethz.ch>
#          Boaz Ko <boazko@student.ethz.ch>
#          Ben Bullinger <ben.bullinger@inf.ethz.ch>
#SBATCH --job-name=orders
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --time=00:25:00
#SBATCH --output=orders.out
cd "${SLURM_SUBMIT_DIR:-$PWD}/src"
ARGS="--nx 128 --ny 128 --nz 64 --num_iter 1024"
for rep in 1 2 3; do
  for n in 2 4 6 8; do
    for lim in "off" "on"; do
      X=""; [ $lim = on ] && X="--limiter"
      echo "### order=$n limiter=$lim rep=$rep"
      srun --uenv=prgenv-gnu/26.3:v1 --view=default -n 72 -c 1 --cpu-bind=cores \
        ./stencil2d-filter.x $ARGS --order $n $X 2>/dev/null | grep -E "^\[|limited"
    done
  done
done
echo ALLDONE

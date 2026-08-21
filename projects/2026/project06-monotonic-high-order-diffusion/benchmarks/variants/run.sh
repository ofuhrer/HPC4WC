#!/bin/bash
#
# Constant-work domain scan over the optimization ladder.
#
# Authors: Stefanie Boersig <stefanie.boersig@env.ethz.ch>
#          Boaz Ko <boazko@student.ethz.ch>
#          Ben Bullinger <ben.bullinger@inf.ethz.ch>
#SBATCH --job-name=variants
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --time=00:28:00
#SBATCH --output=variants.out
cd "${SLURM_SUBMIT_DIR:-$PWD}/src"
L="srun --uenv=prgenv-gnu/26.3:v1 --view=default -n 72 -c 1 --cpu-bind=cores"
bench() { # size iters
  local S=$1 IT=$2
  for rep in 1 2; do
    for v in filter filter-kblock filter-inline; do
      for mode in off lim bfree; do
        X=""; [ $mode = lim ] && X="--limiter"
        [ $mode = bfree ] && { X="--limiter --branchfree"; [ $v = filter ] && continue; }
        for n in 4 8; do
          echo "### v=$v mode=$mode order=$n size=$S iters=$IT rep=$rep"
          $L ./stencil2d-$v.x --nx $S --ny $S --nz 64 --num_iter $IT --order $n $X 2>/dev/null | grep "^\["
        done
      done
    done
  done
}
bench 128 1024
bench 256 256
bench 512 64
bench 1024 16
echo ALLDONE

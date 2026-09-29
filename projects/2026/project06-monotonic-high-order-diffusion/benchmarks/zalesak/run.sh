#!/bin/bash
#
# Zalesak flux correction: runtime and accuracy.
#
# Authors: Stefanie Boersig <stefanie.boersig@env.ethz.ch>
#          Boaz Ko <boazko@student.ethz.ch>
#          Ben Bullinger <ben.bullinger@inf.ethz.ch>
#SBATCH --job-name=zalesak
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --time=00:20:00
#SBATCH --output=zalesak.out
cd "${SLURM_SUBMIT_DIR:-$PWD}/src"
L="srun --uenv=prgenv-gnu/26.3:v1 --view=default -n 72 -c 1 --cpu-bind=cores"
bench() { # size iters
  for rep in 1 2 3; do
    echo "### scheme=off size=$1 rep=$rep"
    $L ./stencil2d-filter.x --nx $1 --ny $1 --nz 64 --num_iter $2 --order 4 2>/dev/null | grep "^\["
    echo "### scheme=simple size=$1 rep=$rep"
    $L ./stencil2d-filter.x --nx $1 --ny $1 --nz 64 --num_iter $2 --order 4 --limiter 2>/dev/null | grep -E "^\[|limited"
    echo "### scheme=zalesak size=$1 rep=$rep"
    $L ./stencil2d-filter-zalesak.x --nx $1 --ny $1 --nz 64 --num_iter $2 --order 4 2>/dev/null | grep -E "^\[|mean_C"
  done
}
bench 128 1024
bench 1024 16
echo ALLDONE

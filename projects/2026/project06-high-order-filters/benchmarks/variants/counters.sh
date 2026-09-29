#!/bin/bash
#
# Hardware counters for the optimization ladder.
#
# Authors: Stefanie Boersig <stefanie.boersig@env.ethz.ch>
#          Boaz Ko <boazko@student.ethz.ch>
#          Ben Bullinger <ben.bullinger@inf.ethz.ch>
#SBATCH --job-name=variants-counters
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --time=00:25:00
#SBATCH --output=variants-counters.out
cd "${SLURM_SUBMIT_DIR:-$PWD}/src"
L="srun --uenv=prgenv-gnu/26.3:v1 --view=default -n 72 -c 1 --cpu-bind=cores"
ARGS="--nx 1024 --ny 1024 --nz 64 --num_iter 256 --order 4"
for v in filter filter-kblock filter-inline; do
  for mode in off lim bfree; do
    X=""; [ $mode = lim ] && X="--limiter"
    [ $mode = bfree ] && { X="--limiter --branchfree"; [ $v = filter ] && continue; }
    rm -rf perf
    echo "### v=$v mode=$mode"
    $L bash ../tools/perf_wrap.sh ./stencil2d-$v.x $ARGS $X 2>/dev/null | grep "^\["
    awk '/mem_access_rd/ {gsub(",","",$1); rd+=$1} /mem_access_wr/ {gsub(",","",$1); wr+=$1} /LLC-load-misses/ {gsub(",","",$1); llc+=$1} END {printf "PERF rd=%d wr=%d llc=%d\n", rd, wr, llc}' perf/*.txt
  done
done
rm -rf perf
echo ALLDONE

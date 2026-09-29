#!/bin/bash
#
# Hardware counters for the Zalesak flux correction.
#
# Authors: Stefanie Boersig <stefanie.boersig@env.ethz.ch>
#          Boaz Ko <boazko@student.ethz.ch>
#          Ben Bullinger <ben.bullinger@inf.ethz.ch>
#SBATCH --job-name=zalesak-counters
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --time=00:20:00
#SBATCH --output=zalesak-counters.out
# Roofline counters for the Zalesak variant, same setup as variants/counters.sh
# (1024^2 x 64, n=4, 256 iterations, 72 ranks on one Grace socket), plus
# same-allocation reruns of the naive filter (off/limited) as baselines.
cd "${SLURM_SUBMIT_DIR:-$PWD}/src"
L="srun --uenv=prgenv-gnu/26.3:v1 --view=default -n 72 -c 1 --cpu-bind=cores"
ARGS="--nx 1024 --ny 1024 --nz 64 --num_iter 256 --order 4"
for run in filter:off filter:lim filter-zalesak:zalesak; do
  v=${run%%:*}; mode=${run##*:}
  X=""; [ $mode = lim ] && X="--limiter"
  rm -rf perf
  echo "### v=$v mode=$mode"
  $L bash ../tools/perf_wrap.sh ./stencil2d-$v.x $ARGS $X 2>/dev/null | grep "^\[\|zalesak_mean_C"
  awk '/mem_access_rd/ {gsub(",","",$1); rd+=$1} /mem_access_wr/ {gsub(",","",$1); wr+=$1} /LLC-load-misses/ {gsub(",","",$1); llc+=$1} END {printf "PERF rd=%d wr=%d llc=%d\n", rd, wr, llc}' perf/*.txt
done
rm -rf perf
echo ALLDONE

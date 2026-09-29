#!/bin/bash
#
# Wraps a command in per-rank perf stat, ARM PMU events on Grace.
#
# Authors: Stefanie Boersig <stefanie.boersig@env.ethz.ch>
#          Boaz Ko <boazko@student.ethz.ch>
#          Ben Bullinger <ben.bullinger@inf.ethz.ch>
rank=$(printf "%05d" "${SLURM_PROCID:-0}")
mkdir -p perf
perf stat -e mem_access_rd:u,mem_access_wr:u,LLC-load-misses:u \
    -o perf/${rank}.txt "$@"

#!/bin/bash

if [ -z "$SLURM_PROCID" ]; then
    echo "Error: SLURM_PROCID is not set. Aborting." >&2
    exit 1
fi

mkdir -p perf

rank=$(printf "%05d" "$SLURM_PROCID")

outf=perf/${rank}.txt

taskset -cp $$ > ${outf} 2>&1

perf stat -e mem_access_rd,mem_access_wr,l2d_cache_refill_rd,l2d_cache_refill_wr "$@" 2>> ${outf}

exit 0

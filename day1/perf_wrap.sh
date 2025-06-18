#!/bin/bash

if [ -z "$SLURM_PROCID" ]; then
    echo "Error: SLURM_PROCID is not set. Aborting." >&2
    exit 1
fi

mkdir -p perf

rank=$(printf "%05d" "$SLURM_PROCID")

outd=perf/${rank}.dat
outf=perf/${rank}.txt

export MPICH_ASYNC_PROGRESS=0
export MPICH_MAX_THREAD_SAFETY=single
export MPIR_CVAR_CH3_NOLOCAL=1
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICH_NEMESIS_ASYNC_PROGRESS=0
export MPICH_ENABLE_HIDDEN_PROGRESS=0
export MPICH_RMA_OVER_DMAPP=0
export MPICH_GNI_NDREG_ENTRIES=0
export MPICH_CH3_NOLOCAL=1

echo "======= taskset ==========" >> ${outf}
taskset -cp $$ > ${outf} 2>&1

echo "======= perf stat ==========" >> ${outf}
perf stat -e mem_access_rd:u,mem_access_wr:u,LLC-load-misses:u "$@" 2>> ${outf}

echo "======= perf record ==========" >> ${outf}
perf record -e cpu-clock:u -o ${outd} "$@" >/dev/null 2>&1
perf report -i ${outd} >> ${outf}

exit 0

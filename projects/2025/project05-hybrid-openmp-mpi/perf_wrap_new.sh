#!/bin/bash

name=""          # Base name or template for files (without extension)
outdir="perf"    # Output directory

while getopts ":n:o:h" opt; do
  case "$opt" in
    n) name="$OPTARG" ;;
    o) outdir="$OPTARG" ;;
    h)
      echo "Usage: $0 [-n name|template] [-o outdir] command [args...]"
      echo "  -n name|template  Base filename (without extension). You can use {rank}."
      echo "  -o outdir         Output directory (default: perf)."
      echo "Example: $0 -n exp-{rank} -o results ./my_prog --foo 1"
      exit 0
      ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done
shift $((OPTIND-1))

# Ensure there is a target command to run under perf
if [ $# -eq 0 ]; then
  echo "Missing target command. Run with -h for help." >&2
  exit 1
fi

# ---- Determine the rank -------------------------------------------------------
if [ -z "${SLURM_PROCID:-}" ] && [ -z "${OMPI_COMM_WORLD_RANK:-}" ]; then
  need_rank="no"
  if [ -z "$name" ]; then
    need_rank="yes"
  elif [[ "$name" == *"{rank}"* ]]; then
    need_rank="yes"
  fi
  if [ "$need_rank" = "yes" ]; then
    echo "Error: SLURM_PROCID and OMPI_COMM_WORLD_RANK are not set. Cannot build a rank-based name." >&2
    exit 1
  fi
fi

# Zero-pad rank to 5 digits
if [ -n "${SLURM_PROCID:-}" ]; then
  rank=$(printf "%05d" "$SLURM_PROCID")
elif [ -n "${OMPI_COMM_WORLD_RANK:-}" ]; then
  rank=$(printf "%05d" "$OMPI_COMM_WORLD_RANK")
else
  rank=""  # not needed if rank not required
fi

# ---- Build output paths -------------------------------------------------------
mkdir -p "$outdir"

if [ -n "$name" ]; then
  base="${name//\{rank\}/$rank}"
else
  base="$rank"
fi

out_dat="$outdir/${base}_${rank}.dat"
out_txt="$outdir/${base}_${rank}.txt"

# ---- Environment tweaks (MPI vars etc.) --------------------------------------
export MPICH_ASYNC_PROGRESS=0
export MPICH_MAX_THREAD_SAFETY=single
export MPIR_CVAR_CH3_NOLOCAL=1
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICH_NEMESIS_ASYNC_PROGRESS=0
export MPICH_ENABLE_HIDDEN_PROGRESS=0
export MPICH_RMA_OVER_DMAPP=0
export MPICH_GNI_NDREG_ENTRIES=0
export MPICH_CH3_NOLOCAL=1

# ---- Perf profiling (only rank 0) --------------------------------------------
# if [ "$rank" = "00000" ]; then

  # Fresh log for rank 0
  # : > ${out_txt}

  echo "======= taskset ==========" >> ${out_txt}
  taskset -cp $$ >> ${out_txt} 2>&1

  echo "======= perf stat ==========" >> ${out_txt}
  perf stat -e mem_access_rd:u,mem_access_wr:u,LLC-load-misses:u,FP_SCALE_OPS_SPEC,FP_FIXED_OPS_SPEC "$@" 2>> ${out_txt}

  echo "======= perf record (functions, rank 0) ==========" >> ${out_txt}
  # Record with call stacks (-F sampling rate, -g for call graph)
    perf record -e cpu-clock:u -o ${out_dat} "$@" >/dev/null 2>&1

  # Check perf record exit status
  if [ $? -ne 0 ]; then
    echo "Error: perf record failed for rank 0" >&2
    exit 1
  fi

  # Report in text mode, sorted by function (self time %)
  perf report --stdio -i ${out_dat} --sort=symbol >> ${out_txt}

# else
#   # All other ranks just run normally
#   "$@"
# fi

exit 0

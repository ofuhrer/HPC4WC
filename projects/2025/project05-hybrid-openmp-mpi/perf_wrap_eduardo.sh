#!/usr/bin/env bash
#
# perf_wrap.sh — Run a target command under `perf` and save both a binary
# perf data file (.dat) and a human-readable log (.txt).
#
# Features:
# - `-n` lets you set the base filename (you can use {rank} as a placeholder).
# - `-o` lets you set the output directory (default: perf).
# - Falls back to the MPI/SLURM rank (zero-padded to 5 digits) if no name is given.
# - Appends to the .txt log (does not overwrite).
#
# Usage:
#   ./perf_wrap.sh [-n name|template] [-o outdir] command [args...]
#
# Examples:
#   ./perf_wrap.sh -n runA ./my_prog --size 1024
#     -> perf/runA.dat, perf/runA.txt
#
#   ./perf_wrap.sh -n exp-{rank} ./my_prog --iters 100
#     -> perf/exp-00007.dat, perf/exp-00007.txt   (if rank=7)
#
#   ./perf_wrap.sh -n run1 -o results ./my_prog
#     -> results/run1.dat, results/run1.txt
#
# Important:
# - Pass wrapper options BEFORE the command you want to profile.
# - Everything after the options is forwarded unchanged to your program.
#

# ---- Parse wrapper options ----------------------------------------------------

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

# ---- Determine the rank (used for default naming and {rank} substitution) -----
# Prefer SLURM_PROCID, then OMPI_COMM_WORLD_RANK. Abort if neither exists when needed.

if [ -z "${SLURM_PROCID:-}" ] && [ -z "${OMPI_COMM_WORLD_RANK:-}" ]; then
  # We only need rank if -n contains {rank} or if no -n was given (default uses rank).
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

# Zero-pad rank to 5 digits (e.g., 7 -> 00007)
if [ -n "${SLURM_PROCID:-}" ]; then
  rank=$(printf "%05d" "$SLURM_PROCID")
elif [ -n "${OMPI_COMM_WORLD_RANK:-}" ]; then
  rank=$(printf "%05d" "$OMPI_COMM_WORLD_RANK")
else
  rank=""  # Not needed; handled above
fi

# ---- Build output paths -------------------------------------------------------

# Create output directory if needed
mkdir -p "$outdir"

# If the user provided a name, substitute {rank} if present; otherwise default to rank
if [ -n "$name" ]; then
  base="${name//\{rank\}/$rank}"
else
  base="$rank"
fi

out_dat="$outdir/${base}.dat"
out_txt="$outdir/${base}.txt"

# ---- Record environment / context into the log --------------------------------
# We append (>>) so re-runs keep accumulating context rather than overwriting.

{
  echo "==================== PERF WRAP ===================="
  date -Is
  echo "Host: $(hostname)"
  echo "CWD : $(pwd)"
  echo "CMD : $*"
  echo "SLURM_PROCID=${SLURM_PROCID:-}"
  echo "OMPI_COMM_WORLD_RANK=${OMPI_COMM_WORLD_RANK:-}"
  echo "Resolved rank: ${rank:-<none>}"
  echo "Output .dat: $out_dat"
  echo "Output .txt: $out_txt"
  echo
  echo "==================== taskset ======================"
} >> "$out_txt"

# Capture CPU affinity of this shell (PID $$)
taskset -cp $$ >> "$out_txt" 2>&1

# ---- Set environment knobs (mirrors original script behavior) -----------------
# Disable various async/progress settings that could affect measurement noise.
export MPICH_ASYNC_PROGRESS=0
export MPICH_MAX_THREAD_SAFETY=single
export MPIR_CVAR_CH3_NOLOCAL=1
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICH_NEMESIS_ASYNC_PROGRESS=0
export MPICH_ENABLE_HIDDEN_PROGRESS=0
export MPICH_RMA_OVER_DMAPP=0
export MPICH_GNI_NDREG_ENTRIES=0
export MPICH_CH3_NOLOCAL=1

# ---- perf stat ---------------------------------------------------------------
{
  echo
  echo "==================== perf stat ===================="
} >> "$out_txt"

# `perf stat` writes its statistics to stderr; append that to the log.
# We let the target's stdout pass through to the console (if any).
perf stat \
  -e mem_access_rd:u,mem_access_wr:u,LLC-load-misses:u,FP_SCALE_OPS_SPEC,FP_FIXED_OPS_SPEC \
  "$@" 2>> "$out_txt"

# ---- perf record + perf report ------------------------------------------------
{
  echo
  echo "==================== perf record =================="
  echo "Recording to: $out_dat"
} >> "$out_txt"

# Suppress perf record stdout noise; errors (if any) will still be visible next lines
perf record -e cpu-clock:u -o "$out_dat" "$@" >/dev/null 2>&1

{
  echo
  echo "==================== perf report =================="
} >> "$out_txt"

# Produce a text summary of the recorded data into the log
perf report -i "$out_dat" >> "$out_txt" 2>&1

# ---- Done --------------------------------------------------------------------
{
  echo
  echo "===================== DONE ========================"
  date -Is
} >> "$out_txt"

exit 0
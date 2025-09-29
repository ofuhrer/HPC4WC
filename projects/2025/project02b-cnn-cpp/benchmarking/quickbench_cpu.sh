#!/usr/bin/env bash
#SBATCH -J quickbench_cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH -t 00:30:00
#SBATCH --hint=nomultithread
#SBATCH -o benchmarking/logs/%x-%j.out

set -euo pipefail

# -------------------------
# Parse flags (same as quickbench_cpu.py)
#   --samples INT
#   --repeats INT
#   --threads <list...>
#   --outdir PATH
# Also allow 2 positionals: repeats, samples
# -------------------------
REPEATS=2
SAMPLES=10000
OUTDIR=""
THREADS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeats)
      REPEATS="$2"; shift 2 ;;
    --samples)
      SAMPLES="$2"; shift 2 ;;
    --outdir)
      OUTDIR="$2"; shift 2 ;;
    --threads)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        THREADS+=("$1")
        shift
      done
      ;;
    *)
      # positionals: repeats then samples
      if [[ -z "${_pos1:-}" ]]; then REPEATS="$1"; _pos1=1
      elif [[ -z "${_pos2:-}" ]]; then SAMPLES="$1"; _pos2=1
      fi
      shift ;;
  esac
done

mkdir -p benchmarking/logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "[info] SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK:-unset}"
echo "[info] repeats=${REPEATS}, samples=${SAMPLES}, threads=${THREADS[*]:-default}, outdir=${OUTDIR:-<auto>}"

# -------------------------
# Modules & build (no clean)
# -------------------------
module purge
module load stack/2024-06 gcc/12.2.0
module load cmake
module load libtorch/2.1.0

cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU=OFF
cmake --build build-cpu -j "${SLURM_CPUS_PER_TASK:-8}"

# -------------------------
# Baseline threading policy
# (Per-run OMP_NUM_THREADS is set inside quickbench_cpu.py)
# -------------------------
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_SCHEDULE=static

# prevent hidden threading in math libs
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -------------------------
# Choose script
# -------------------------
SCRIPT="benchmarking/quickbench_cpu.py"
if [[ ! -f "$SCRIPT" ]]; then
  echo "[error] $SCRIPT not found." >&2
  exit 1
fi

# -------------------------
# Build argv for quickbench
# -------------------------
argv=( "$SCRIPT" "--repeats" "$REPEATS" "--samples" "$SAMPLES" )
if [[ -n "$OUTDIR" ]]; then
  argv+=( "--outdir" "$OUTDIR" )
fi
if [[ ${#THREADS[@]} -gt 0 ]]; then
  argv+=( "--threads" "${THREADS[@]}" )
fi

# -------------------------
# Run under Slurm step with core binding
# -------------------------
srun -n 1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
     --cpu-bind=cores --hint=nomultithread \
     python3 "${argv[@]}"

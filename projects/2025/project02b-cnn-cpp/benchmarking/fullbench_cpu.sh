#!/usr/bin/env bash
#SBATCH -J fullbench_cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH -t 08:00:00
#SBATCH --hint=nomultithread
#SBATCH -o benchmarking/logs/%x-%j.out


set -euo pipefail

# -------------------------
# Inputs (positional or flags)
# -------------------------
REPEATS=10
SAMPLES=50000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeats)
      REPEATS="$2"; shift 2 ;;
    --samples)
      SAMPLES="$2"; shift 2 ;;
    *)
      if [[ -z "${_pos1:-}" ]]; then REPEATS="$1"; _pos1=1
      elif [[ -z "${_pos2:-}" ]]; then SAMPLES="$1"; _pos2=1
      fi
      shift ;;
  esac
done

# -------------------------
# Housekeeping
# -------------------------
mkdir -p benchmarking/logs
cd "$SLURM_SUBMIT_DIR"

echo "[info] SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK:-unset}"
echo "[info] repeats=${REPEATS}, samples=${SAMPLES}"

# -------------------------
# Modules & build
# -------------------------
module purge
module load stack/2024-06 gcc/12.2.0
module load cmake
module load libtorch/2.1.0

rm -rf build-cpu
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU=OFF
cmake --build build-cpu -j "${SLURM_CPUS_PER_TASK:-16}"

# -------------------------
# Python env
# -------------------------
# If this is a virtualenv:
source ~/cnn-bench/bin/activate

# -------------------------
# Run benchmark (ensure core binding)
# -------------------------
# Prefer the canonical filename; fall back if the colleague’s name is present.
if [[ -f benchmarking/fullbenchmark_cpu.py ]]; then
  SCRIPT="benchmarking/fullbenchmark_cpu.py"
else
  echo "[error] fullbench script not found." >&2
  exit 1
fi

# --- Added: OpenMP and math-lib threading controls for stability ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_SCHEDULE=static

# prevent nested / hidden threading in BLAS and friends
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# -------------------------------------------------------------------

# Run under Slurm step with core binding so child processes inherit the cpuset
srun -n 1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
     --cpu-bind=cores --hint=nomultithread \
     python3 "$SCRIPT" --repeats "${REPEATS}" --samples "${SAMPLES}"

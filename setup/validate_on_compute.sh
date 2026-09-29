#!/usr/bin/env bash
#SBATCH --job-name=hpc4wc-validate
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --view=default

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC4WC_ROOT="${HPC4WC_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
HPC4WC_LOG_DIR="${HPC4WC_LOG_DIR:-${SCRATCH:-${HPC4WC_ROOT}}/HPC4WC_validation_logs}"
HPC4WC_VENV_DIR="${HPC4WC_VENV_DIR:-${HOME}/HPC4WC_venv}"

mkdir -p "${HPC4WC_LOG_DIR}"
LOG_FILE="${HPC4WC_LOG_DIR}/validate-${SLURM_JOB_ID:-manual}-$(date +%Y%m%dT%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1

echo "==> HPC4WC compute-node validation"
echo "root: ${HPC4WC_ROOT}"
echo "log: ${LOG_FILE}"
echo "host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<not in Slurm>}"
echo "SLURM_NTASKS: ${SLURM_NTASKS:-<unset>}"

if [ -f "${HPC4WC_VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${HPC4WC_VENV_DIR}/bin/activate"
else
    echo "ERROR: Cannot find venv at ${HPC4WC_VENV_DIR}." >&2
    echo "Run setup/HPC4WC_setup.sh from JupyterHub first, or set HPC4WC_VENV_DIR." >&2
    exit 1
fi

python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
VENV_SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="${VENV_SITE_PACKAGES}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${VIRTUAL_ENV}/bin/python"

echo
echo "==> uenv and toolchain"
uenv status || true
which python
python --version
echo "validation python: ${PYTHON_BIN}"
which mpicc || true
which mpif90 || true
which nvcc || true
nvidia-smi

echo
echo "==> Single-process validation without GT4Py GPU"
"${PYTHON_BIN}" -u "${HPC4WC_ROOT}/setup/validate_environment.py" --require-gpu --skip-gt4py-gpu

echo
echo "==> Single-process GT4Py GPU validation"
"${PYTHON_BIN}" -u "${HPC4WC_ROOT}/setup/validate_environment.py" \
    --require-gpu \
    --require-gt4py-gpu \
    --only-gt4py-gpu

echo
echo "==> MPI validation via srun"
srun -n "${SLURM_NTASKS:-4}" -c 1 \
    "${PYTHON_BIN}" -u "${HPC4WC_ROOT}/setup/validate_environment.py" \
    --mpi-only \
    --require-mpi-size "${SLURM_NTASKS:-4}"

echo
echo "Validation completed successfully."

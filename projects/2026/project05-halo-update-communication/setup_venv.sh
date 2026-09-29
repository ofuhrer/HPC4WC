#!/usr/bin/env bash
# Create (or recreate) the local .venv for this repo on Santis.
# Run this from a Santis compute node (JupyterHub terminal), after the
# prgenv-gnu uenv is active, so mpicc/mpif90 are on PATH.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

if [ -e "${VENV_DIR}" ]; then
    echo "ERROR: ${VENV_DIR} already exists. Remove it first if you want to recreate it." >&2
    exit 1
fi

echo "==> Creating virtual environment at ${VENV_DIR}"
python -m venv --system-site-packages "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Installing packages from requirements.txt"
python -m pip install --upgrade pip
export MPICC="${MPICC:-$(command -v mpicc)}"
python -m pip install --no-binary=mpi4py -r "${REPO_ROOT}/requirements.txt"

echo "==> Done. Activate with: source ${VENV_DIR}/bin/activate"

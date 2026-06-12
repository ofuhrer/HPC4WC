#!/usr/bin/env bash
set -euo pipefail

TARGET_UENV_NAME="prgenv-gnu"
TARGET_UENV_VERSION="26.3"
TARGET_UENV="${TARGET_UENV_NAME}/${TARGET_UENV_VERSION}:v1"

HPC4WC_ROOT="${HPC4WC_ROOT:-${HOME}/HPC4WC}"
HPC4WC_KERNEL_NAME="${HPC4WC_KERNEL_NAME:-HPC4WC_kernel}"
HPC4WC_KERNEL_DISPLAY_NAME="${HPC4WC_KERNEL_DISPLAY_NAME:-HPC4WC_kernel}"
HPC4WC_FORCE="${HPC4WC_FORCE:-0}"

if [ -z "${SCRATCH:-}" ]; then
    echo "ERROR: SCRATCH is not set. Start this from the CSCS JupyterHub/Santis environment." >&2
    exit 1
fi

HPC4WC_VENV_DIR="${HPC4WC_VENV_DIR:-${SCRATCH}/HPC4WC_venv-${TARGET_UENV_NAME}-${TARGET_UENV_VERSION}}"
HPC4WC_VENV_LINK="${HPC4WC_VENV_LINK:-${HOME}/HPC4WC_venv}"
HPC4WC_CACHE_DIR="${HPC4WC_CACHE_DIR:-${SCRATCH}/.cache/hpc4wc}"

log() {
    echo "==> $*"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

log "Checking repository and Santis uenv environment"
[ -d "${HPC4WC_ROOT}" ] || die "Cannot find HPC4WC repository at ${HPC4WC_ROOT}"
[ -f "${HPC4WC_ROOT}/setup/etc/requirements.txt" ] || die "Missing setup/etc/requirements.txt"

require_command uenv
require_command python
require_command gcc
require_command gfortran
require_command mpicc
require_command mpif90
require_command nvcc

uenv status || true
if ! uenv status 2>/dev/null | grep -q "${TARGET_UENV_NAME}:/user-environment"; then
    die "The ${TARGET_UENV_NAME} uenv is not active. Launch JupyterHub with ${TARGET_UENV}."
fi

if [ -r /user-environment/meta/configure.json ]; then
    if ! grep -q "${TARGET_UENV_NAME}/${TARGET_UENV_VERSION}" /user-environment/meta/configure.json; then
        if [ "${HPC4WC_ALLOW_UENV_MISMATCH:-0}" != "1" ]; then
            die "Active uenv does not look like ${TARGET_UENV}. Set HPC4WC_ALLOW_UENV_MISMATCH=1 to override."
        fi
        echo "WARNING: Active uenv does not look like ${TARGET_UENV}; continuing because override is set." >&2
    fi
fi

python - <<'PY'
import sys

version = sys.version_info
if version < (3, 12):
    raise SystemExit(f"Python >= 3.12 is required, found {sys.version}")
print(f"Python: {sys.version.split()[0]}")
PY

log "Toolchain summary"
echo "python: $(command -v python) ($(python --version 2>&1))"
echo "gcc:    $(command -v gcc) ($(gcc --version | head -1))"
echo "gfortran: $(command -v gfortran) ($(gfortran --version | head -1))"
echo "mpicc:  $(command -v mpicc)"
echo "mpif90: $(command -v mpif90)"
echo "nvcc:   $(command -v nvcc)"
nvcc --version | tail -4 || true

if [ -e "${HPC4WC_VENV_DIR}" ] || [ -L "${HPC4WC_VENV_LINK}" ] || [ -e "${HPC4WC_VENV_LINK}" ]; then
    if [ "${HPC4WC_FORCE}" != "1" ]; then
        die "HPC4WC venv already exists. Set HPC4WC_FORCE=1 to recreate it."
    fi
    log "Removing existing HPC4WC venv because HPC4WC_FORCE=1"
    rm -rf "${HPC4WC_VENV_DIR}"
    rm -rf "${HPC4WC_VENV_LINK}"
fi

log "Creating cache directories under ${HPC4WC_CACHE_DIR}"
mkdir -p \
    "${HPC4WC_CACHE_DIR}/pip" \
    "${HPC4WC_CACHE_DIR}/cupy" \
    "${HPC4WC_CACHE_DIR}/gt4py" \
    "$(dirname "${HPC4WC_VENV_DIR}")"

export XDG_CACHE_HOME="${HPC4WC_CACHE_DIR}"
export PIP_CACHE_DIR="${HPC4WC_CACHE_DIR}/pip"
export CUPY_CACHE_DIR="${HPC4WC_CACHE_DIR}/cupy"
export GT4PY_CACHE_DIR="${HPC4WC_CACHE_DIR}/gt4py"

log "Creating Python virtual environment at ${HPC4WC_VENV_DIR}"
python -m venv --system-site-packages "${HPC4WC_VENV_DIR}"
# shellcheck disable=SC1091
source "${HPC4WC_VENV_DIR}/bin/activate"

VENV_SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="${VENV_SITE_PACKAGES}:${PYTHONPATH:-}"

log "Installing Python packages"
python -m pip install --upgrade pip
python -m pip install setuptools wheel
export MPICC="${MPICC:-$(command -v mpicc)}"
python -m pip install --no-binary=mpi4py -r "${HPC4WC_ROOT}/setup/etc/requirements.txt"

log "Creating compatibility symlink ${HPC4WC_VENV_LINK}"
ln -sfn "${HPC4WC_VENV_DIR}" "${HPC4WC_VENV_LINK}"

log "Creating ${HPC4WC_KERNEL_NAME} Jupyter kernel"
python -m ipykernel install \
    --user \
    --name="${HPC4WC_KERNEL_NAME}" \
    --display-name="${HPC4WC_KERNEL_DISPLAY_NAME}" \
    --env PATH "${PATH}" \
    --env VIRTUAL_ENV "${VIRTUAL_ENV}" \
    --env PYTHONPATH "${PYTHONPATH}" \
    --env XDG_CACHE_HOME "${XDG_CACHE_HOME}" \
    --env PIP_CACHE_DIR "${PIP_CACHE_DIR}" \
    --env CUPY_CACHE_DIR "${CUPY_CACHE_DIR}" \
    --env GT4PY_CACHE_DIR "${GT4PY_CACHE_DIR}"

if [ ! -d "${HOME}/.local/share/jupyter/kernels/${HPC4WC_KERNEL_NAME,,}" ]; then
    die "Problem installing the Jupyter kernel."
fi

log "Installing Bash kernel"
python -m bash_kernel.install

log "Writing terminal activation helper"
cat > "${HOME}/activate_hpc4wc.sh" <<EOF
#!/usr/bin/env bash
source "${HPC4WC_VENV_DIR}/bin/activate"
export XDG_CACHE_HOME="${XDG_CACHE_HOME}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR}"
export GT4PY_CACHE_DIR="${GT4PY_CACHE_DIR}"
export PYTHONPATH="\$(python -c 'import site; print(site.getsitepackages()[0])'):\${PYTHONPATH:-}"
EOF
chmod +x "${HOME}/activate_hpc4wc.sh"

log "Validating CPU-side environment"
python "${HPC4WC_ROOT}/setup/validate_environment.py" --skip-gpu

log "Successfully finished. Restart your JupyterHub server, then select ${HPC4WC_KERNEL_DISPLAY_NAME}."
echo "For terminals, run: source ${HOME}/activate_hpc4wc.sh"

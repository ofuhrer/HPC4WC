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

remove_stale_srun_wrapper() {
    local candidate
    for candidate in "${HPC4WC_VENV_DIR}/bin/srun" "${HPC4WC_VENV_LINK}/bin/srun"; do
        if [ -f "${candidate}" ] && grep -q "spawner-jupyterhub" "${candidate}"; then
            log "Removing stale HPC4WC srun wrapper at ${candidate}"
            rm -f "${candidate}"
        fi
    done
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

uenv_status="$(uenv status 2>/dev/null || true)"
printf "%s\n" "${uenv_status}"
if ! printf "%s\n" "${uenv_status}" | grep -q "${TARGET_UENV_NAME}:/user-environment" \
    && ! { printf "%s\n" "${uenv_status}" | grep -Eq "^uenv[[:space:]]+${TARGET_UENV_NAME}([[:space:]]|$)" \
        && printf "%s\n" "${uenv_status}" | grep -Eq "^[[:space:]]*mount[[:space:]]+/user-environment([[:space:]]|$)"; }; then
    die "The ${TARGET_UENV_NAME} uenv is not active. Launch JupyterHub with ${TARGET_UENV}."
fi

if [ -r /user-environment/meta/configure.json ]; then
    if ! grep -q "${TARGET_UENV_NAME}/${TARGET_UENV_VERSION}" /user-environment/meta/configure.json; then
        echo "WARNING: Could not confirm exact uenv version ${TARGET_UENV} from /user-environment metadata." >&2
        echo "WARNING: Continuing because ${TARGET_UENV_NAME} is active and the required tools are available." >&2
    fi
fi

python - <<'PY'
import sys

version = sys.version_info
if version[:2] != (3, 14):
    raise SystemExit(
        "Python 3.14 from prgenv-gnu/26.3:v1 is required, "
        f"found {sys.version.split()[0]}. "
        "Stop this JupyterHub server and relaunch it with prgenv-gnu/26.3:v1."
    )
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
if ! nvcc --version | grep -q "release 13.1"; then
    die "CUDA 13.1 from ${TARGET_UENV} is required. Stop this JupyterHub server and relaunch it with ${TARGET_UENV}."
fi

if [ -e "${HPC4WC_VENV_DIR}" ] || [ -L "${HPC4WC_VENV_LINK}" ] || [ -e "${HPC4WC_VENV_LINK}" ]; then
    remove_stale_srun_wrapper
    if [ "${HPC4WC_FORCE}" != "1" ]; then
        die "HPC4WC venv already exists. Set HPC4WC_FORCE=1 to recreate it."
    fi
    log "Removing existing HPC4WC venv because HPC4WC_FORCE=1"
    rm -rf "${HPC4WC_VENV_DIR}"
    rm -rf "${HPC4WC_VENV_LINK}"
fi

mkdir -p "$(dirname "${HPC4WC_VENV_DIR}")"

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

log "Using Santis CMake for GT4Py builds"
python -m pip uninstall -y cmake >/dev/null 2>&1 || true
rm -f "${VIRTUAL_ENV}/bin/cmake" "${VIRTUAL_ENV}/bin/cpack" "${VIRTUAL_ENV}/bin/ctest"
hash -r
require_command cmake
require_command ninja
echo "cmake: $(command -v cmake) ($(cmake --version | head -1))"
echo "ninja: $(command -v ninja) ($(ninja --version))"

log "Creating compatibility symlink ${HPC4WC_VENV_LINK}"
ln -sfn "${HPC4WC_VENV_DIR}" "${HPC4WC_VENV_LINK}"

KERNEL_LAUNCHER="${VIRTUAL_ENV}/bin/hpc4wc_kernel_launcher.py"
log "Installing ${HPC4WC_KERNEL_NAME} kernel launcher"
cat > "${KERNEL_LAUNCHER}" <<'EOF'
#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import sys


def main() -> None:
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
    os.execv(sys.executable, [sys.executable, "-m", "ipykernel_launcher", *sys.argv[1:]])


if __name__ == "__main__":
    main()
EOF
chmod +x "${KERNEL_LAUNCHER}"

log "Creating ${HPC4WC_KERNEL_NAME} Jupyter kernel"
python -m ipykernel install \
    --user \
    --name="${HPC4WC_KERNEL_NAME}" \
    --display-name="${HPC4WC_KERNEL_DISPLAY_NAME}" \
    --env PATH "${PATH}" \
    --env VIRTUAL_ENV "${VIRTUAL_ENV}" \
    --env PYTHONPATH "${PYTHONPATH}"

if [ ! -d "${HOME}/.local/share/jupyter/kernels/${HPC4WC_KERNEL_NAME,,}" ]; then
    die "Problem installing the Jupyter kernel."
fi

KERNEL_JSON="${HOME}/.local/share/jupyter/kernels/${HPC4WC_KERNEL_NAME,,}/kernel.json"
python - "${KERNEL_JSON}" "${VIRTUAL_ENV}/bin/python" "${KERNEL_LAUNCHER}" <<'PY'
import json
import sys
from pathlib import Path

kernel_json = Path(sys.argv[1])
python_bin = sys.argv[2]
launcher = sys.argv[3]

spec = json.loads(kernel_json.read_text())
spec["argv"] = [python_bin, launcher, "-f", "{connection_file}"]
kernel_json.write_text(json.dumps(spec, indent=2) + "\n")
PY

log "Installing Bash kernel"
python -m bash_kernel.install

BASH_KERNEL_JSON="${HOME}/.local/share/jupyter/kernels/bash/kernel.json"
[ -f "${BASH_KERNEL_JSON}" ] || die "Problem installing the Bash kernel."

log "Configuring Bash kernel environment"
python - "${BASH_KERNEL_JSON}" "${VIRTUAL_ENV}/bin/python" "${PATH}" "${VIRTUAL_ENV}" "${PYTHONPATH}" <<'PY'
import json
import sys
from pathlib import Path

kernel_json = Path(sys.argv[1])
python_bin = sys.argv[2]
path = sys.argv[3]
virtual_env = sys.argv[4]
pythonpath = sys.argv[5]

spec = json.loads(kernel_json.read_text())
spec["argv"] = [python_bin, "-m", "bash_kernel", "-f", "{connection_file}"]
env = spec.setdefault("env", {})
env["PATH"] = path
env["VIRTUAL_ENV"] = virtual_env
env["PYTHONPATH"] = pythonpath
kernel_json.write_text(json.dumps(spec, indent=2) + "\n")
PY

log "Writing terminal activation helper"
cat > "${HOME}/activate_hpc4wc.sh" <<EOF
#!/usr/bin/env bash
source "${HPC4WC_VENV_DIR}/bin/activate"
export PYTHONPATH="\$(python -c 'import site; print(site.getsitepackages()[0])'):\${PYTHONPATH:-}"
EOF
chmod +x "${HOME}/activate_hpc4wc.sh"

log "Validating CPU-side environment"
python "${HPC4WC_ROOT}/setup/validate_environment.py" --skip-gpu

log "Successfully finished. Restart your JupyterHub server, then select ${HPC4WC_KERNEL_DISPLAY_NAME}."
echo "For terminals, run: source ${HOME}/activate_hpc4wc.sh"

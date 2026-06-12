#!/usr/bin/env python3
"""Validate the HPC4WC Santis/JupyterHub Python environment."""

from __future__ import annotations

import argparse
import builtins
import importlib.metadata
import os
import platform
import shutil
import socket
import subprocess
import sys
import warnings
from collections.abc import Callable
from functools import partial

import numpy as np

print = partial(builtins.print, flush=True)

EXPECTED_PACKAGES = {
    "gt4py": "1.1.10",
    "cupy-cuda13x": "14.1.1",
    "mpi4py": "4.1.2",
    "numpy": "2.4.6",
    "matplotlib": "3.11.0",
    "ipykernel": "7.3.0",
    "ipyparallel": "9.2.0",
    "ipcmagic-cscs": "1.1.0",
    "bash_kernel": "0.10.0",
}

try:
    import gt4py.next as gtx

    I = gtx.Dimension("I")
    J = gtx.Dimension("J")
    K = gtx.Dimension("K")

    IJKField = gtx.Field[gtx.Dims[I, J, K], gtx.float64]

    @gtx.field_operator
    def add_one(inp: IJKField) -> IJKField:
        return inp + 1.0

    @gtx.program
    def add_one_program(
        inp: IJKField,
        out: IJKField,
        nx: gtx.int32,
        ny: gtx.int32,
        nz: gtx.int32,
    ):
        add_one(inp, out=out, domain={I: (0, nx), J: (0, ny), K: (0, nz)})

except Exception:
    gtx = None


class CheckFailed(RuntimeError):
    """Raised when an environment check fails."""


def section(title: str) -> None:
    print(f"\n== {title} ==")


def run_check(name: str, check: Callable[[], None]) -> bool:
    print(f"[ RUN ] {name}")
    try:
        check()
    except Exception as exc:
        print(f"[FAIL ] {name}: {exc}")
        return False
    print(f"[ OK  ] {name}")
    return True


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable ({exc})"


def check_summary() -> None:
    print(f"hostname: {socket.gethostname()}")
    print(f"machine: {platform.machine()}")
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    print(f"SHELL: {os.environ.get('SHELL', '<unset>')}")
    print(f"HOME: {os.environ.get('HOME', '<unset>')}")
    print(f"SCRATCH: {os.environ.get('SCRATCH', '<unset>')}")
    print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', '<unset>')}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '<unset>')}")
    print(f"uenv status:\n{command_output(['uenv', 'status'])}")

    for command in ("gcc", "gfortran", "mpicc", "mpif90", "nvcc", "nvidia-smi"):
        path = shutil.which(command)
        print(f"{command}: {path or '<not found>'}")


def check_package_versions() -> None:
    mismatches = []
    for package, expected_version in EXPECTED_PACKAGES.items():
        try:
            actual_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{package}: not installed, expected {expected_version}")
            continue
        print(f"{package}: {actual_version}")
        if actual_version != expected_version:
            mismatches.append(f"{package}: {actual_version}, expected {expected_version}")

    if mismatches:
        raise CheckFailed("; ".join(mismatches))


def check_numpy_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.arange(9.0).reshape(3, 3)
    result = matrix @ matrix.T
    if not np.allclose(result.diagonal(), [5.0, 50.0, 149.0]):
        raise CheckFailed("unexpected NumPy matmul result")

    fig, ax = plt.subplots()
    ax.imshow(result)
    fig.canvas.draw()
    plt.close(fig)


def check_mpi(required_size: int | None) -> None:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"MPI rank {rank} of {size} on {socket.gethostname()}")
    if required_size is not None and size != required_size:
        raise CheckFailed(f"MPI size is {size}, expected {required_size}")


def mpi_rank_size() -> tuple[int, int]:
    try:
        from mpi4py import MPI
    except Exception:
        return 0, 1

    comm = MPI.COMM_WORLD
    return comm.Get_rank(), comm.Get_size()


def cupy_device_count() -> int:
    import cupy as cp

    return int(cp.cuda.runtime.getDeviceCount())


def check_cupy(skip_gpu: bool, require_gpu: bool) -> bool:
    if skip_gpu:
        print("CuPy GPU check skipped")
        return False

    import cupy as cp

    device_count = cupy_device_count()
    print(f"CUDA devices: {device_count}")
    if device_count < 1:
        if require_gpu:
            raise CheckFailed("no CUDA devices visible")
        print("No CUDA devices visible; skipping CuPy allocation")
        return False

    values = cp.arange(1024, dtype=cp.float64)
    result = float(cp.sum(values).get())
    cp.cuda.Stream.null.synchronize()
    if result != 523776.0:
        raise CheckFailed(f"unexpected CuPy reduction result: {result}")
    return True


def _run_gt4py_backend(backend: object, label: str) -> None:
    if gtx is None:
        raise CheckFailed("gt4py.next could not be imported")

    nx, ny, nz = 4, 3, 2
    domain = gtx.domain({I: (0, nx), J: (0, ny), K: (0, nz)})
    data = np.arange(nx * ny * nz, dtype=np.float64).reshape(nx, ny, nz)
    in_field = gtx.as_field(domain, data, allocator=backend)
    out_field = gtx.zeros(domain, dtype=gtx.float64, allocator=backend)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Field View Program 'add_one_program': Using Python execution.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Python is not running in optimized mode.*",
            category=UserWarning,
        )
        add_one_program.with_backend(backend)(
            in_field,
            out_field,
            nx,
            ny,
            nz,
            offset_provider={},
        )
    if backend is gtx.gtfn_gpu:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()

    if not np.allclose(out_field.asnumpy(), data + 1.0):
        raise CheckFailed(f"unexpected GT4Py result for {label}")
    print(f"GT4Py {label}: ok")


def check_gt4py(
    skip_gpu: bool,
    skip_gt4py_gpu: bool,
    only_gt4py_gpu: bool,
    require_gpu: bool,
    require_gt4py_gpu: bool,
) -> None:
    if gtx is None:
        raise CheckFailed("gt4py.next could not be imported")

    if not only_gt4py_gpu:
        _run_gt4py_backend(None, "None")
        _run_gt4py_backend(gtx.gtfn_cpu, "gtx.gtfn_cpu")

    if skip_gpu or skip_gt4py_gpu:
        print("GT4Py GPU backend skipped")
        return

    has_gpu = cupy_device_count() > 0
    if not has_gpu:
        if require_gpu or require_gt4py_gpu:
            raise CheckFailed("no CUDA devices visible for GT4Py GPU backend")
        print("No CUDA devices visible; skipping GT4Py GPU backend")
        return

    if not only_gt4py_gpu:
        print("Starting GT4Py GPU backend in a separate Python process")
        subprocess.run(
            [
                sys.executable,
                "-u",
                __file__,
                "--require-gpu",
                "--require-gt4py-gpu",
                "--only-gt4py-gpu",
            ],
            check=True,
        )
        return

    _run_gt4py_backend(gtx.gtfn_gpu, "gtx.gtfn_gpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpi-only", action="store_true", help="only run the mpi4py check")
    parser.add_argument("--skip-gpu", action="store_true", help="skip CUDA/CuPy/GT4Py GPU checks")
    parser.add_argument(
        "--skip-gt4py-gpu",
        action="store_true",
        help="run CUDA/CuPy checks but skip the GT4Py GPU backend",
    )
    parser.add_argument(
        "--only-gt4py-gpu",
        action="store_true",
        help="only run the GT4Py GPU backend in the GT4Py check",
    )
    parser.add_argument("--require-gpu", action="store_true", help="fail if no CUDA GPU is visible")
    parser.add_argument(
        "--require-gt4py-gpu",
        action="store_true",
        help="fail if the GT4Py GPU backend cannot run",
    )
    parser.add_argument(
        "--require-mpi-size",
        type=int,
        default=None,
        help="fail if MPI_COMM_WORLD does not have this size",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = []
    mpi_rank, mpi_size = mpi_rank_size()

    if args.mpi_only:
        section("MPI")
        if run_check("mpi4py", lambda: check_mpi(args.require_mpi_size)):
            return 0
        return 1

    if mpi_size > 1 and mpi_rank != 0:
        section("MPI")
        if run_check("mpi4py", lambda: check_mpi(args.require_mpi_size)):
            return 0
        return 1

    section("Environment summary")
    results.append(run_check("environment summary", check_summary))

    section("Python packages")
    results.append(run_check("pinned package versions", check_package_versions))

    section("CPU Python stack")
    results.append(run_check("NumPy and Matplotlib", check_numpy_matplotlib))

    section("MPI")
    results.append(run_check("mpi4py", lambda: check_mpi(args.require_mpi_size)))

    section("GPU stack")
    results.append(run_check("CuPy GPU", lambda: check_cupy(args.skip_gpu, args.require_gpu)))

    section("GT4Py")
    results.append(
        run_check(
            "GT4Py backends",
            lambda: check_gt4py(
                skip_gpu=args.skip_gpu,
                skip_gt4py_gpu=args.skip_gt4py_gpu,
                only_gt4py_gpu=args.only_gt4py_gpu,
                require_gpu=args.require_gpu,
                require_gt4py_gpu=args.require_gt4py_gpu,
            ),
        )
    )

    if all(results):
        print("\nAll requested environment checks passed.")
        return 0

    print("\nOne or more environment checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

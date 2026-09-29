"""Build and run a variant's compiled Fortran column driver.

No numpy re-implementation of any physics lives here -- this module only builds the
Fortran binary and shuttles CSVs in and out of it, so every value it returns comes
straight from the reference scheme running natively. That is the whole point of the
consistency harnesses: the Fortran is the reference, not a numpy transcription of it.

Every variant lays its driver out the same way (``<variant>/fortran/Makefile`` with a
``driver`` target producing ``build/column_driver``), so the only per-variant input is
the path to that ``fortran/`` directory.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from mcrph_common.csv_io import read_columns, write_columns


def driver_exe(fortran_dir: str | Path) -> Path:
    return Path(fortran_dir) / "build" / "column_driver"


def ensure_driver_built(fortran_dir: str | Path) -> None:
    """Build ``build/column_driver`` via ``make driver`` (idempotent).

    Honours an optional ``FC`` override (e.g. ``FC=gfortran-16``) for machines whose
    default ``gfortran`` is missing or too old.
    """
    cmd = ["make", "driver"]
    fc = os.environ.get("FC")
    if fc:
        cmd.append(f"FC={fc}")
    proc = subprocess.run(cmd, cwd=str(fortran_dir), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to build the Fortran driver with `{' '.join(cmd)}` "
            f"(cwd={fortran_dir}):\n{proc.stdout}\n{proc.stderr}"
        )


def run_driver(
    fortran_dir: str | Path,
    inputs: Mapping[str, np.ndarray],
    extra_args: Sequence[str] = (),
) -> dict[str, np.ndarray]:
    """Run one column through the compiled driver and return its output columns.

    ``inputs`` is written as the driver's input CSV in mapping order, so it must
    match the column order the driver's ``column_driver.f90`` expects. ``extra_args``
    are appended after the input/output paths (satad's ``tol``/``maxiter``, say).

    A throwaway temp directory (stdlib :mod:`tempfile`, OS temp dir) holds the input
    and output CSVs, so the committed ``example/`` files are never touched by a run.
    """
    exe = driver_exe(fortran_dir)
    if not exe.exists():
        raise RuntimeError(f"driver not built ({exe}); call ensure_driver_built() first")

    with tempfile.TemporaryDirectory() as tmp:
        in_csv = os.path.join(tmp, "in.csv")
        out_csv = os.path.join(tmp, "out.csv")
        write_columns(in_csv, inputs)

        proc = subprocess.run(
            [str(exe), in_csv, out_csv, *extra_args], capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"column_driver failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
            )

        return read_columns(out_csv)

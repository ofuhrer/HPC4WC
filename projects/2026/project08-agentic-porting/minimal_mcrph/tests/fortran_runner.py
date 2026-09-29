"""Drive the *real* compiled minimal_mcrph Fortran column driver.

The build/run plumbing is variant-agnostic and lives in
:mod:`mcrph_common.fortran_runner`; all this module adds is this variant's column
layout, its second input file (``hhl.csv``), and its optional per-stage dumps. No numpy
re-implementation of the physics lives here or there -- every value returned comes
straight from the reference scheme running natively, which is the whole point: the
Fortran is the reference, not a transcription of it.

Contrast with what this replaces. The previous tests asserted against 9-12 digit
literals pasted from instrumented runs that were then reverted; they could not be
regenerated, and the transcription alone put a floor under every tolerance. Running the
binary costs a few hundred milliseconds and removes that entire category of problem.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from mcrph_common.csv_io import read_columns, write_columns
from mcrph_common.fortran_runner import driver_exe
from mcrph_common.fortran_runner import ensure_driver_built as _ensure_built
from mcrph_common.fortran_runner import run_driver
from minimal_mcrph import FORTRAN_DIR
from minimal_mcrph.csv_io import COLUMNS as FIELD_HEADER

# Boundaries mo_stage_dump.f90 writes, in pipeline order. Same names on both sides, so a
# stage needs no translation table -- see minimal_mcrph/driver.py::Driver._STAGE_FIELDS.
STAGES = (
    "satad_pre",
    "prepare",
    "ccn",
    "default_n",
    "ice_nuc",
    "cloud_freeze",
    "vapor_dep",
    "ice_melt",
    "post",
)


def ensure_driver_built() -> None:
    """Build ``build/column_driver`` via ``make driver`` (idempotent)."""
    _ensure_built(FORTRAN_DIR)


def run_fortran(fields, hhl, dt=30.0, want_stages=False):
    """Run one column through the compiled Fortran and return its output columns.

    ``fields`` maps column name -> 1-D array and must cover every name in
    ``FIELD_HEADER``; ``hhl`` is the nlev+1 half-level heights. Returns
    ``{column: array}`` for the final state, or ``(final, stages)`` when
    ``want_stages`` is set, where ``stages`` maps each name in :data:`STAGES` to its own
    ``{column: array}``.

    ``mcrph_common.fortran_runner.run_driver`` writes exactly one input CSV, but this
    driver needs ``hhl.csv`` as well and can be asked for a dump directory, so both are
    placed in a temp directory here and handed over as positional arguments. That keeps
    the shared helper unchanged and this variant's extra plumbing local.
    """
    columns = {
        name: np.asarray(fields[name], dtype=np.float64).ravel() for name in FIELD_HEADER
    }
    hhl = np.asarray(hhl, dtype=np.float64).ravel()
    nlev = len(columns["rho"])
    if hhl.shape[0] != nlev + 1:
        raise ValueError(f"hhl has {hhl.shape[0]} rows, expected nlev+1={nlev + 1}")

    exe = driver_exe(FORTRAN_DIR)
    if not exe.exists():
        raise RuntimeError(f"driver not built ({exe}); call ensure_driver_built() first")

    with tempfile.TemporaryDirectory() as tmp:
        hhl_csv = os.path.join(tmp, "hhl.csv")
        write_columns(hhl_csv, {"hhl": hhl})

        extra = [hhl_csv, repr(float(dt))]
        if want_stages:
            extra.append(tmp)

        final = run_driver(FORTRAN_DIR, columns, extra_args=extra)

        if not want_stages:
            return final

        stages = {name: read_columns(os.path.join(tmp, f"{name}.csv")) for name in STAGES}
        return final, stages


def run_fortran_coeffs():
    """Return ``{particle: {coeff: value}}`` from the Fortran's one-time coefficient setup.

    These are pure functions of the static particle constants, so the column used to
    provoke the run is irrelevant; the bundled example is used for convenience.
    """
    from minimal_mcrph import EXAMPLE_DIR
    from minimal_mcrph.csv_io import read_hhl_csv

    fields = read_columns(EXAMPLE_DIR / "fields.csv")
    hhl = read_hhl_csv(EXAMPLE_DIR / "hhl.csv")

    exe = driver_exe(FORTRAN_DIR)
    if not exe.exists():
        raise RuntimeError(f"driver not built ({exe}); call ensure_driver_built() first")

    with tempfile.TemporaryDirectory() as tmp:
        hhl_csv = os.path.join(tmp, "hhl.csv")
        write_columns(hhl_csv, {"hhl": hhl})
        run_driver(
            FORTRAN_DIR,
            {name: fields[name] for name in FIELD_HEADER},
            extra_args=[hhl_csv, "30.0", tmp],
        )
        with open(os.path.join(tmp, "coeffs.csv")) as fh:
            names = [n.strip() for n in fh.readline().split(",")]
            out = {}
            for line in fh:
                cells = [c.strip() for c in line.split(",")]
                out[cells[0]] = {n: float(v) for n, v in zip(names[1:], cells[1:])}
    return out

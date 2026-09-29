"""Drive the *real* compiled Fortran satad column driver.

The build/run plumbing itself is variant-agnostic and lives in
:mod:`mcrph_common.fortran_runner`; all this module adds is satad's column layout
(``rho,tk,qv,qc``) and its iteration controls. No numpy re-implementation of the
physics lives here or there -- every value returned comes straight from
``mo_satad.f90`` running natively.
"""

from __future__ import annotations

import numpy as np

from mcrph_common.fortran_runner import ensure_driver_built as _ensure_built
from mcrph_common.fortran_runner import run_driver
from satad_only import FORTRAN_DIR

FIELD_HEADER = ("rho", "tk", "qv", "qc")


def ensure_driver_built() -> None:
    """Build satad's ``build/column_driver`` via ``make driver`` (idempotent)."""
    _ensure_built(FORTRAN_DIR)


def run_fortran(rho, tk, qv, qc, tol, maxiter):
    """Run saturation adjustment on one column through the compiled Fortran.

    Parameters follow the CSV column order (``rho, tk, qv, qc``) plus the iteration
    controls. Returns the adjusted ``(tk, qv, qc)`` as float64 numpy arrays with the
    same length as the (1-D) inputs; ``rho`` is unchanged by the physics.
    """
    columns = {
        name: np.asarray(values, dtype=np.float64).ravel()
        for name, values in zip(FIELD_HEADER, (rho, tk, qv, qc))
    }
    out = run_driver(
        FORTRAN_DIR, columns, extra_args=[repr(float(tol)), str(int(maxiter))]
    )
    return out["tk"], out["qv"], out["qc"]

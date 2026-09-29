"""Shared fixtures and helpers for the native satad consistency harness.

Bridges to the *real* GT4Py port (embedded DSL execution) and ensures the *real*
Fortran driver is built once per session. Nothing here re-implements the physics.
"""

from __future__ import annotations

import pytest

from mcrph_common.compare import compare as _compare
from satad_only.satad_gt4py import DEFAULT_TOL, MAXITER
from satad_only.tests.fortran_runner import ensure_driver_built

# Iteration controls shared by both implementations. GT4Py fixes these at import
# time (MAXITER unrolled, DEFAULT_TOL); the Fortran is told the same so the two
# solve an identical problem.
TOL = DEFAULT_TOL
NEWTON_MAXITER = MAXITER

# Comparison tolerance (user spec): relative only, no absolute floor -- except
# for fields computed around zero.
RTOL = 10e-12
QC_ATOL = 1e-18  # qc floors at ZQWMIN=1e-20 / 0.0 -> needs a tiny absolute floor


@pytest.fixture(scope="session", autouse=True)
def _built_driver():
    """Build the Fortran driver before any test runs."""
    ensure_driver_built()


def compare(name, got, ref, rtol=RTOL, atol=0.0):
    """satad's default tolerances applied to the shared comparison helper."""
    _compare(name, got, ref, rtol=rtol, atol=atol)

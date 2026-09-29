"""Shared fixtures and tolerances for the minimal_mcrph consistency harness.

Bridges to the *real* GT4Py port and ensures the *real* Fortran driver is built once per
session. Nothing here re-implements the physics.

TOLERANCES
----------
The numbers below are measurements, not aspirations. After three transcription bugs were
found and fixed (see PORT_NOTES.md: single-precision particle literals, satad's
early-exit condition, and the ``ddust_background`` scale factor), the port reproduces the
Fortran to a few ULP on every field of every scenario -- many of them bit-for-bit. RTOL
is therefore set just above the observed worst case rather than at some round number that
happens to pass.

This matters more than it sounds. The previous harness asserted ``rtol=1e-6``, which is
nine orders of magnitude looser than what the port achieves, and that slack was exactly
what let a real 5e-7 error in the depositional-growth coefficients sit undetected. A
tolerance far above the achievable floor is not "safe"; it is a test that has stopped
testing.
"""

from __future__ import annotations

import numpy as np
import pytest

from mcrph_common.compare import compare as _compare
from minimal_mcrph.tests.fortran_runner import ensure_driver_built

# Worst observed relative deviation across all six scenarios, every field, every stage
# boundary: 1.638e-15 (~7 ULP). `warm` comes out bit-identical. What remains is expression
# association -- e.g. the Fortran writes `b * (c * sqrt(D*v))` where the port uses the
# precomputed `b_f * sqrt(D*v)` -- which no amount of care will remove without rewriting
# one side to match the other's parenthesisation.
#
# 1e-14 leaves roughly 6x headroom over that: tight enough that any real regression fails,
# loose enough that a different libm or a reassociating compiler does not.
RTOL = 1.0e-14

# Absolute floors, needed only where a field is driven to (or clipped at) exactly zero,
# so that a relative comparison has no meaningful denominator. Everything else gets
# atol=0 and a pure relative check.
#
#   qc          floors at ZQWMIN=1e-20 or exact 0.0 in satad's evaporation branch
#   q*/qn*      frozen species are clipped to exact 0.0 by the driver's negative clip
#   ninact      relaxes toward zero when qi == 0
#
# Each floor is set well below the smallest *physically meaningful* value of its field,
# so it can only ever absorb noise around zero, never a real difference.
ATOL = {
    "qc": 1.0e-25,
    "qr": 1.0e-25,
    "qi": 1.0e-25,
    "qs": 1.0e-25,
    "qg": 1.0e-25,
    "qh": 1.0e-25,
    "qnc": 1.0e-10,
    "qnr": 1.0e-10,
    "qni": 1.0e-10,
    "qns": 1.0e-10,
    "qng": 1.0e-10,
    "qnh": 1.0e-10,
    "ninact": 1.0e-10,
    "ninpot": 1.0e-10,
    "nccn": 1.0e-10,
}

# Fields the port carries through a timestep. `pres`, `w` and `rho` are inputs the scheme
# never modifies, and `ssat`/`ninagi`/`qrsflux` are inert in this configuration.
CHECKED_FIELDS = (
    "tk", "qv", "qc", "qnc", "qr", "qnr", "qi", "qni",
    "qs", "qns", "qg", "qng", "qh", "qnh", "nccn", "ninpot", "ninact",
)  # fmt: skip


@pytest.fixture(scope="session", autouse=True)
def _built_driver():
    """Build the Fortran driver before any test runs."""
    ensure_driver_built()


def compare(name, got, ref, rtol=RTOL, atol=0.0):
    """minimal_mcrph's default tolerances applied to the shared comparison helper."""
    _compare(name, got, ref, rtol=rtol, atol=atol)


def compare_field(label, field, got, ref, rtol=RTOL):
    """Compare one named field, picking up its absolute floor from :data:`ATOL`."""
    compare(f"[{label}] {field}", got, ref, rtol=rtol, atol=ATOL.get(field, 0.0))


def compare_columns(label, got, ref, fields=CHECKED_FIELDS, rtol=RTOL):
    """Compare a whole ``{name: array}`` column state field by field.

    Every field is checked before anything is raised, so a failure reports all the fields
    that moved rather than only the first -- which is the difference between "the ice
    physics drifted" and "something is off somewhere".
    """
    failures = []
    for field in fields:
        try:
            compare_field(label, field, got[field], ref[field], rtol=rtol)
        except AssertionError as exc:  # noqa: PERF203
            failures.append(str(exc))
    if failures:
        raise AssertionError(
            f"{len(failures)} of {len(fields)} fields differ in [{label}]:\n\n"
            + "\n\n".join(failures)
        )


def as_columns(state, fields=CHECKED_FIELDS):
    """Pull ``{name: array}`` off a :class:`~minimal_mcrph.driver.ColumnState`."""
    return {name: np.asarray(getattr(state, name), dtype=np.float64) for name in fields}

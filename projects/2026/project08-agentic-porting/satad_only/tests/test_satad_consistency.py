"""Native Fortran <-> GT4Py consistency tests for saturation adjustment.

For every scenario column the SAME inputs are pushed through both the compiled
Fortran ``satad_v_3D`` (via ``fortran_runner``) and the GT4Py port (via
``satad_numpy``), and the adjusted ``(tk, qv, qc)`` are required to agree. No
numpy transcription of the Fortran is used as an oracle -- the reference is the
real Fortran binary. Both sides are given identical ``tol``/``maxiter`` so they
solve the same problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from satad_only.satad_gt4py import satad_numpy
from satad_only.tests.conftest import compare, TOL, NEWTON_MAXITER, QC_ATOL
from satad_only.tests.fortran_runner import run_fortran
from satad_only.tests.scenarios import SCENARIOS

_NAMES = sorted(SCENARIOS)


@pytest.mark.parametrize("name", _NAMES)
def test_fortran_gt4py_consistency(name):
    rho, tk, qv, qc = SCENARIOS[name]

    tk_f, qv_f, qc_f = run_fortran(rho, tk, qv, qc, tol=TOL, maxiter=NEWTON_MAXITER)
    tk_g, qv_g, qc_g = satad_numpy(rho, tk, qv, qc, tol=TOL)

    compare(f"[{name}] tk", tk_g, tk_f)
    compare(f"[{name}] qv", qv_g, qv_f)
    # qc floors at 0.0 / ZQWMIN and can land near zero, where a pure relative
    # tolerance is unusable -> allow a tiny absolute floor here only.
    compare(f"[{name}] qc", qc_g, qc_f, atol=QC_ATOL)


@pytest.mark.parametrize("name", _NAMES)
def test_physical_invariants(name):
    """Oracle-free checks on the GT4Py output: water conservation, non-negative
    cloud, and the sign of the temperature change vs condensation/evaporation.
    """
    rho, tk, qv, qc = (np.asarray(x, dtype=np.float64) for x in SCENARIOS[name])
    tk_g, qv_g, qc_g = satad_numpy(rho, tk, qv, qc, tol=TOL)

    # adjustable water qv+qc is conserved by the adjustment
    assert np.allclose(qv_g + qc_g, qv + qc, rtol=0, atol=1e-12)
    # cloud water never goes negative
    assert np.all(qc_g >= 0.0)
    # condensation (qc up) warms; evaporation (qc down) cools
    dqc = qc_g - qc
    dtk = tk_g - tk
    assert np.all(dtk[dqc > 1e-15] > 0.0)
    assert np.all(dtk[dqc < -1e-15] < 0.0)

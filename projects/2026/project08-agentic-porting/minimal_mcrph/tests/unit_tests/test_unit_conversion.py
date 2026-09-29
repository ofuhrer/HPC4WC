"""unit_conversion.py's algebraic properties.

Scope note: the numerical agreement of the density conversions and the latent-heat
temperature update with the Fortran is covered by ``test_stage_boundaries.py`` at the
``prepare`` and ``post`` boundaries, where they are checked in place against a live
Fortran run over six columns. The tests here are deliberately *not* Fortran-referenced --
they pin properties that follow from the code itself (exact round-tripping, clipping
behaviour) and would still hold if the reference changed.

Stating that plainly matters: an earlier version of this file claimed Fortran provenance
in its docstring for tests that were partly hand-derived, which makes it easy to believe
coverage exists where it does not.

Every test runs under both the embedded and the compiled backend. That is not padding:
a plain module-level Python float referenced inside a ``field_operator`` body resolves
fine under embedded execution and fails to compile under ``gtfn_cpu``, so the compiled
run is what actually verifies the ``_Const`` enum wiring described in
``unit_conversion.py``.
"""

from __future__ import annotations

import gt4py.next as gtx
import numpy as np
import pytest
from icon4py.model.common import dimension as dims

from minimal_mcrph.stencils import unit_conversion as uc

BACKENDS = [None, gtx.gtfn_cpu]


def _field(values, backend):
    return gtx.as_field(
        [dims.CellDim, dims.KDim],
        np.array(values, dtype=np.float64).reshape(1, -1),
        allocator=backend,
    )


@pytest.mark.parametrize("backend", BACKENDS, ids=["embedded", "gtfn_cpu"])
def test_clip_negative(backend):
    program = uc.clip_negative.with_backend(backend) if backend is not None else uc.clip_negative
    q = _field([-1.0, -1e-30, 0.0, 5.0], backend)
    program(q, 0, 1, 0, 4, offset_provider={})
    # Exact: the clip either passes a value through untouched or writes a literal 0.0.
    assert list(q.asnumpy().flatten()) == [0.0, 0.0, 0.0, 5.0]


@pytest.mark.parametrize("backend", BACKENDS, ids=["embedded", "gtfn_cpu"])
def test_convert_fields_round_trip(backend):
    """Converting to densities and back must return the original values exactly.

    Chosen so it can be asserted exactly rather than approximately: multiplying by 2.0 and
    then by 0.5 is lossless in binary floating point, so any deviation at all means the
    conversion is not applying the factor it claims (wrong field, wrong direction, or a
    field silently skipped).
    """
    original = [1.0, 2.0, 3.0]
    q = _field(original, backend)
    rho = _field([2.0, 2.0, 2.0], backend)
    rho_r = _field([0.5, 0.5, 0.5], backend)

    uc.convert_fields(
        [q], rho, horizontal_start=0, horizontal_end=1,
        vertical_start=0, vertical_end=3, backend=backend,
    )  # fmt: skip
    assert list(q.asnumpy().flatten()) == [2.0, 4.0, 6.0]

    uc.convert_fields(
        [q], rho_r, horizontal_start=0, horizontal_end=1,
        vertical_start=0, vertical_end=3, backend=backend,
    )  # fmt: skip
    assert list(q.asnumpy().flatten()) == original


@pytest.mark.parametrize("backend", BACKENDS, ids=["embedded", "gtfn_cpu"])
def test_density_corrections_are_monotone_in_rho(backend):
    """Both corrections fall as air gets denser, and equal 1 at their reference density.

    A property check rather than a value check: ``rhocorr`` and ``rhocld`` are negative
    powers of ``rho/rho_0``, so they must decrease monotonically and cross 1.0 at
    ``rho = rho_0 = 1.225``. Getting the sign of the exponent wrong -- the plausible
    error -- inverts the ordering and is caught here without any reference data.
    """
    program = (
        uc.compute_density_corrections.with_backend(backend)
        if backend is not None
        else uc.compute_density_corrections
    )
    rho_values = [0.4, 0.8, 1.225, 1.6]
    rho = _field(rho_values, backend)
    rhocorr = _field([0.0] * 4, backend)
    rhocld = _field([0.0] * 4, backend)
    program(rho, rhocorr, rhocld, 0, 1, 0, 4, offset_provider={})

    for name, out in (("rhocorr", rhocorr), ("rhocld", rhocld)):
        values = out.asnumpy().flatten()
        assert np.all(np.diff(values) < 0.0), f"{name} must decrease with density"
        np.testing.assert_allclose(values[2], 1.0, rtol=1e-14,
                                   err_msg=f"{name} must be 1.0 at rho_0")  # fmt: skip

"""set_default_n and the size clips, checked against their closed-form definitions.

Scope note: these are **not** Fortran-referenced, and that is deliberate rather than a
gap. The expected values below are evaluated directly from the ``set_qnc``/``set_qni``/
``set_qnr``/``set_qns``/``set_qng``/``set_qnh_expPSD_N0const`` formulas in
``mo_2mom_mcrph_util.f90``, which are closed-form expressions with no branching beyond a
``q >= 1e-20`` guard. Numerical agreement of these stencils inside the real pipeline is
covered against a live Fortran run by ``test_stage_boundaries.py`` at the ``default_n``
boundary; what is left for here is confirming each species uses *its own* formula and
constants, which a whole-column comparison would miss whenever the species in question is
absent from the column.

The previous version of this file gestured at Fortran validation in its docstring while
hand-deriving the numbers, and asserted at ``rtol=1e-6``. Both are corrected: the
provenance is stated plainly, and since the expectations are exact closed forms evaluated
in the same double precision as the stencil, the tolerance is 1e-14. A looser bound here
would not be caution, it would just stop detecting a wrong constant.

Note the cloud branch is guarded in Fortran by whether the optional ``n_cn`` argument is
present -- but ``clouds_twomoment``'s one call site (mo_2mom_mcrph_main.f90:584) never
passes it, so the cloud branch is always active for this scheme.
"""

import math

import gt4py.next as gtx
import numpy as np
import pytest
from icon4py.model.common import dimension as dims

from minimal_mcrph.stencils import housekeeping as hk

BACKENDS = [None, gtx.gtfn_cpu]


def _field(values, backend):
    return gtx.as_field(
        [dims.CellDim, dims.KDim],
        np.array(values, dtype=np.float64).reshape(1, -1),
        allocator=backend,
    )


PI = math.pi
RHO_W = 1000.0


@pytest.mark.parametrize("backend", BACKENDS, ids=["embedded", "gtfn_cpu"])
def test_set_default_n(backend):
    program = hk.set_default_n.with_backend(backend) if backend is not None else hk.set_default_n

    # level 0: q>0, n=0 (below eps) -> formula fires; level 1: q=0 -> untouched
    cloud_q, cloud_n = _field([1e-4, 0.0], backend), _field([0.0, 5.0], backend)
    ice_q, ice_n = _field([1e-6, 0.0], backend), _field([0.0, 5.0], backend)
    rain_q, rain_n = _field([1e-5, 0.0], backend), _field([0.0, 5.0], backend)
    snow_q, snow_n = _field([1e-5, 0.0], backend), _field([0.0, 5.0], backend)
    graupel_q, graupel_n = _field([1e-4, 0.0], backend), _field([0.0, 5.0], backend)
    hail_q, hail_n = _field([1e-4, 0.0], backend), _field([0.0, 5.0], backend)

    program(
        cloud_q, cloud_n, ice_q, ice_n, rain_q, rain_n,
        snow_q, snow_n, graupel_q, graupel_n, hail_q, hail_n,
        0, 1, 0, 2, offset_provider={},
    )  # fmt: skip

    dmean = 10e-6
    expected_cloud_n = 1e-4 * 6.0 / (PI * RHO_W * dmean**3)
    np.testing.assert_allclose(cloud_n.asnumpy().flatten(), [expected_cloud_n, 5.0], rtol=1e-14)

    expected_ice_n = 1e-6 / 1e-10
    np.testing.assert_allclose(ice_n.asnumpy().flatten(), [expected_ice_n, 5.0], rtol=1e-14)

    n0r, gamma4 = 8000.0e3, 6.0
    expected_rain_n = n0r * (1e-5 * 6.0 / (PI * RHO_W * n0r * gamma4)) ** 0.25
    np.testing.assert_allclose(rain_n.asnumpy().flatten(), [expected_rain_n, 5.0], rtol=1e-14)

    n0s, ams, bms, gamma_bms1 = 800.0e3, 0.038, 2.0, 2.0
    expected_snow_n = n0s * (1e-5 / (ams * n0s * gamma_bms1)) ** (1.0 / (1.0 + bms))
    np.testing.assert_allclose(snow_n.asnumpy().flatten(), [expected_snow_n, 5.0], rtol=1e-14)

    n0g, amg, bmg, gamma_bmg1 = 4000.0e3, 169.6, 3.1, math.gamma(4.1)
    expected_graupel_n = n0g * (1e-4 / (amg * n0g * gamma_bmg1)) ** (1.0 / (1.0 + bmg))
    np.testing.assert_allclose(graupel_n.asnumpy().flatten(), [expected_graupel_n, 5.0], rtol=1e-14)

    rhobulk, n0h = 750.0, 1.0e6
    expected_hail_n = n0h * (1e-4 / (PI * rhobulk * n0h)) ** 0.25
    np.testing.assert_allclose(hail_n.asnumpy().flatten(), [expected_hail_n, 5.0], rtol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS, ids=["embedded", "gtfn_cpu"])
def test_clip_number_concentration(backend):
    program = (
        hk.clip_number_concentration.with_backend(backend)
        if backend is not None
        else hk.clip_number_concentration
    )
    x_min, x_max = 1e-15, 1e-9
    q = _field([1e-4, 1e-4, 1e-4], backend)
    n = _field([1e20, 1e-3, 5e7], backend)  # too high / too low / within bounds
    program(q, n, x_min, x_max, 0, 1, 0, 3, offset_provider={})
    np.testing.assert_allclose(n.asnumpy().flatten(), [1e-4 / x_min, 1e-4 / x_max, 5e7], rtol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS, ids=["embedded", "gtfn_cpu"])
def test_clip_cloud_hard_cap(backend):
    program = (
        hk.clip_cloud_hard_cap.with_backend(backend) if backend is not None else hk.clip_cloud_hard_cap
    )
    n = _field([6000.0e6, 1000.0e6], backend)
    program(n, 0, 1, 0, 2, offset_provider={})
    np.testing.assert_allclose(n.asnumpy().flatten(), [5000.0e6, 1000.0e6])

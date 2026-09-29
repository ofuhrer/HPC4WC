"""``set_default_n`` and number-concentration clipping.

Mirrors ``mo_2mom_mcrph_processes.f90::set_default_n`` (lines 2065-2116), using
the ``mo_2mom_mcrph_util.f90`` ``set_qnc``/``set_qni``/``set_qnr``/``set_qns``/
``set_qng``/``set_qnh_expPSD_N0const`` formulas, plus the
``clip_number_concentration`` housekeeping steps interleaved through
``clouds_twomoment`` (``mo_2mom_mcrph_main.f90:557-651``).

Note: ``set_default_n``'s cloud branch is guarded in Fortran by whether the
optional ``n_cn`` argument is present, but the one call site in
``clouds_twomoment`` (line 584) never passes it -- so for this scheme the cloud
branch is unconditionally active, same as the other five species. Don't add an
``n_cn``-based guard here; it would never fire in the Fortran either.
"""

import enum

import gt4py.next as gtx
from gt4py.next import exp, log, maximum, minimum, where

from icon4py.model.common import dimension as dims
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common import type_alias as ta


class _HousekeepingConst(ta.wpfloat, enum.Enum):
    """See unit_conversion.py's ``_UnitConversionConst`` docstring for why these must be
    enum members rather than plain module floats."""

    EPS_N = 1.0e-3  # set_default_n's threshold on n (mo_2mom_mcrph_processes.f90:2077)
    Q_GUARD = 1.0e-20  # guards log(0) in set_qnr/qns/qng/qnh_expPSD_N0const

    PI = 3.14159265358979323846
    RHO_W = 1000.0  # rhoh2o

    QNC_DMEAN_CUBED = (10.0e-6) ** 3.0  # set_qnc: Dmean=10um cloud droplet

    QNR_N0 = 8000.0e3  # set_qnr: intercept of MP distribution
    QNR_GAMMA4 = 6.0  # Gamma(4.0), exact

    QNS_N0 = 800.0e3  # set_qns
    QNS_AMS = 0.038
    QNS_BMS = 2.0
    QNS_GAMMA_BMS_PLUS1 = 2.0  # Gamma(bms+1=3.0), exact

    QNG_N0 = 4000.0e3  # set_qng
    QNG_AMG = 169.6
    QNG_BMG = 3.1
    QNG_GAMMA_BMG_PLUS1 = 6.812622863016675  # Gamma(bmg+1=4.1), math.gamma(4.1)

    QNH_RHOBULK = 750.0  # set_qnh_expPSD_N0const's assumed bulk density of hail
    QNH_N0 = 1.0e6  # assumed constant N0 of the exponential size distribution


# -- util.f90 set_qnc/set_qni/set_qnr/set_qns/set_qng/set_qnh_expPSD_N0const --


@gtx.field_operator
def _set_qnc(qc: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    return qc * 6.0 / (_HousekeepingConst.PI * _HousekeepingConst.RHO_W * _HousekeepingConst.QNC_DMEAN_CUBED)


@gtx.field_operator
def _set_qni(qi: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    return qi / 1.0e-10


@gtx.field_operator
def _set_qnr(qr: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    unguarded = _HousekeepingConst.QNR_N0 * exp(
        log(qr * 6.0 / (_HousekeepingConst.PI * _HousekeepingConst.RHO_W * _HousekeepingConst.QNR_N0 * _HousekeepingConst.QNR_GAMMA4)) * 0.25
    )
    return where(qr >= _HousekeepingConst.Q_GUARD, unguarded, 0.0)


@gtx.field_operator
def _set_qns(qs: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    unguarded = _HousekeepingConst.QNS_N0 * exp(
        log(qs / (_HousekeepingConst.QNS_AMS * _HousekeepingConst.QNS_N0 * _HousekeepingConst.QNS_GAMMA_BMS_PLUS1))
        * (1.0 / (1.0 + _HousekeepingConst.QNS_BMS))
    )
    return where(qs >= _HousekeepingConst.Q_GUARD, unguarded, 0.0)


@gtx.field_operator
def _set_qng(qg: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    unguarded = _HousekeepingConst.QNG_N0 * exp(
        log(qg / (_HousekeepingConst.QNG_AMG * _HousekeepingConst.QNG_N0 * _HousekeepingConst.QNG_GAMMA_BMG_PLUS1))
        * (1.0 / (1.0 + _HousekeepingConst.QNG_BMG))
    )
    return where(qg >= _HousekeepingConst.Q_GUARD, unguarded, 0.0)


@gtx.field_operator
def _set_qnh(qh: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    unguarded = _HousekeepingConst.QNH_N0 * exp(
        log(qh / (_HousekeepingConst.PI * _HousekeepingConst.QNH_RHOBULK * _HousekeepingConst.QNH_N0)) * 0.25
    )
    return where(qh >= _HousekeepingConst.Q_GUARD, unguarded, 0.0)


# -- set_default_n itself: fills n where q>0 and n<eps, per species --


@gtx.field_operator
def _default_n_cloud(
    q: fa.CellKField[ta.wpfloat], n: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    return where((q > 0.0) & (n < _HousekeepingConst.EPS_N), _set_qnc(q), n)


@gtx.field_operator
def _default_n_ice(
    q: fa.CellKField[ta.wpfloat], n: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    return where((q > 0.0) & (n < _HousekeepingConst.EPS_N), _set_qni(q), n)


@gtx.field_operator
def _default_n_rain(
    q: fa.CellKField[ta.wpfloat], n: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    return where((q > 0.0) & (n < _HousekeepingConst.EPS_N), _set_qnr(q), n)


@gtx.field_operator
def _default_n_snow(
    q: fa.CellKField[ta.wpfloat], n: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    return where((q > 0.0) & (n < _HousekeepingConst.EPS_N), _set_qns(q), n)


@gtx.field_operator
def _default_n_graupel(
    q: fa.CellKField[ta.wpfloat], n: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    return where((q > 0.0) & (n < _HousekeepingConst.EPS_N), _set_qng(q), n)


@gtx.field_operator
def _default_n_hail(
    q: fa.CellKField[ta.wpfloat], n: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    return where((q > 0.0) & (n < _HousekeepingConst.EPS_N), _set_qnh(q), n)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def set_default_n(
    cloud_q: fa.CellKField[ta.wpfloat],
    cloud_n: fa.CellKField[ta.wpfloat],
    ice_q: fa.CellKField[ta.wpfloat],
    ice_n: fa.CellKField[ta.wpfloat],
    rain_q: fa.CellKField[ta.wpfloat],
    rain_n: fa.CellKField[ta.wpfloat],
    snow_q: fa.CellKField[ta.wpfloat],
    snow_n: fa.CellKField[ta.wpfloat],
    graupel_q: fa.CellKField[ta.wpfloat],
    graupel_n: fa.CellKField[ta.wpfloat],
    hail_q: fa.CellKField[ta.wpfloat],
    hail_n: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _default_n_cloud(
        cloud_q,
        cloud_n,
        out=cloud_n,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _default_n_ice(
        ice_q,
        ice_n,
        out=ice_n,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _default_n_rain(
        rain_q,
        rain_n,
        out=rain_n,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _default_n_snow(
        snow_q,
        snow_n,
        out=snow_n,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _default_n_graupel(
        graupel_q,
        graupel_n,
        out=graupel_n,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _default_n_hail(
        hail_q,
        hail_n,
        out=hail_n,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# -- clip_number_concentration: n clamped to [q/x_max, q/x_min] --
# (mo_2mom_mcrph_main.f90:587-590, 603-606, 633-649; both textual orderings --
# max-then-min or min-then-max -- give the same result since x_min < x_max
# always holds, so q/x_max <= q/x_min for q>=0: a simple clamp.)


@gtx.field_operator
def _clip_number_concentration(
    q: fa.CellKField[ta.wpfloat],
    n: fa.CellKField[ta.wpfloat],
    x_min: ta.wpfloat,
    x_max: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    return minimum(maximum(n, q / x_max), q / x_min)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def clip_number_concentration(
    q: fa.CellKField[ta.wpfloat],
    n: fa.CellKField[ta.wpfloat],
    x_min: ta.wpfloat,
    x_max: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _clip_number_concentration(
        q,
        n,
        x_min,
        x_max,
        out=n,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# -- cloud number concentration hard cap (mo_2mom_mcrph_main.f90:637) --
# Only applied to cloud, only in the final clipping step, only if nuc_c_typ>0
# (true for this scheme's nuc_c_typ=8) -- driver's job to call this only there.


@gtx.field_operator
def _clip_cloud_hard_cap(n: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    return minimum(n, 5000.0e6)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def clip_cloud_hard_cap(
    n: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _clip_cloud_hard_cap(
        n,
        out=n,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# -- zero n where q is negligible (mo_2mom_prepare.f90:155-162) --
# Prevents a stale/leftover n from surviving when q has been clipped to ~0.


@gtx.field_operator
def _zero_n_where_q_tiny(
    q: fa.CellKField[ta.wpfloat], n: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    return where(q <= 1.0e-12, 0.0, n)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def zero_n_where_q_tiny(
    q: fa.CellKField[ta.wpfloat],
    n: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _zero_n_where_q_tiny(
        q,
        n,
        out=n,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

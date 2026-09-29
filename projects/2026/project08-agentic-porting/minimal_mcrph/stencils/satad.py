"""satad_v_3D: saturation adjustment (mo_satad.f90:107-282).

Each level is fully independent (no vertical coupling at all -- `rhotot(k)`,
`te(k)`, `qve(k)`, `qce(k)` only ever reference level k), so despite the
Newton iteration this is an ordinary pointwise stencil, not something needing
`scan_operator` or icon4py's grid machinery. Ported directly here rather than
wiring up icon4py's `SaturationAdjustment` component (the plan's original
suggestion) -- that class expects an `IconGrid`/`VerticalGrid`, real
infrastructure for the full unstructured-mesh model that a single CSV-driven
column has no natural instance of. Same physics (icon4py's own Newton
iteration in `saturation_adjustment_stencils.py` uses the same `qsat_rho`/
`dqsatdT_rho` names, confirming this is the same algorithm), simpler to
stand up for one column.

Fortran's Newton loop exits early: `DO WHILE (ABS(twork-tworkold) > tol .AND.
count < maxiter)` with `tol = 1e-3` K and `maxiter = 10`, so a level stops
iterating as soon as one step moves it by less than `tol`. This port carries
that condition as a mask instead of a loop bound (see `_newton_step`).

An earlier version ran all 10 iterations unconditionally, on the argument that
further steps at a converged fixed point are idempotent. They very nearly are --
but not bitwise: the Fortran stops a step or two *short* of the fixed point,
the unconditional version walks all the way to it, and the two therefore land
on different doubles. Measured against the real Fortran, that choice alone cost
2.6e-13 relative on temperature, 5.8e-12 on `qv` and 2.1e-11 on `qc` (which is
a difference of near-equal terms and so amplifies it). Since satad runs first,
that error floor propagated into every field downstream and put a ~1e-11 floor
under the whole scheme's validation. Masking costs nothing -- all 10 steps are
computed either way, the mask only decides which results are kept -- and makes
the port reproduce the Fortran exactly.
"""

import enum

import gt4py.next as gtx
from gt4py.next import abs, maximum, where  # noqa: A004

from icon4py.model.common import dimension as dims
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common import type_alias as ta

from minimal_mcrph.stencils.saturation import e_ws


class _SatadConst(ta.wpfloat, enum.Enum):
    """See unit_conversion.py's ``_UnitConversionConst`` docstring for why
    these must be enum members rather than plain module floats."""

    T_MELT = 273.15
    R_V = 461.51  # water vapor gas constant
    TETENS_BW = 35.86  # c4les
    TETENS_DER = 17.269 * (273.15 - 35.86)  # c5les = c3les*(tmelt-c4les)
    CVD = 1004.64 - 287.04  # cpd - rd
    L_VAPORIZATION = 2.5008e6  # alv
    CP_V = 1850.0  # specific heat of water vapor
    CLW = (3.1733 + 1.0) * 1004.64  # specific heat of liquid water, rcpl=3.1733
    Q_MIN = 1.0e-20  # zqwmin
    # Newton exit threshold, passed as `tol` at both satad_v_3d call sites in
    # mo_nwp_gscp_interface.f90. MAXITER below is the matching `maxiter`, and is the
    # number of times _newton_step is unrolled -- the two must be changed together.
    TOL = 1.0e-3


MAXITER = 10


@gtx.field_operator
def _qsat_rho(
    temperature: fa.CellKField[ta.wpfloat], rho: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    """mo_satad.f90:510-527."""
    return e_ws(temperature) / (rho * _SatadConst.R_V * temperature)


@gtx.field_operator
def _dqsatdt_rho(
    qsat: fa.CellKField[ta.wpfloat], temperature: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    """mo_satad.f90:533-563, ipsat=1 branch."""
    beta = _SatadConst.TETENS_DER / (
        (temperature - _SatadConst.TETENS_BW) * (temperature - _SatadConst.TETENS_BW)
    ) - 1.0 / temperature
    return beta * qsat


@gtx.field_operator
def _latent_heat_vaporization(temperature: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    """mo_satad.f90:612-628."""
    return (
        _SatadConst.L_VAPORIZATION
        + (_SatadConst.CP_V - _SatadConst.CLW) * (temperature - _SatadConst.T_MELT)
        - _SatadConst.R_V * temperature
    )


@gtx.field_operator
def _newton_step(
    twork: fa.CellKField[ta.wpfloat],
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    lwdocvd: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """One unmasked Newton step, mo_satad.f90:249-257."""
    qwd = _qsat_rho(twork, rho)
    dqwd = _dqsatdt_rho(qwd, twork)
    return twork - (twork - temperature + lwdocvd * (qwd - qv)) / (1.0 + lwdocvd * dqwd)


@gtx.field_operator
def satad(
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    lwdocvd = _latent_heat_vaporization(temperature) / _SatadConst.CVD
    t_test = temperature - lwdocvd * qc
    q_test = _qsat_rho(t_test, rho)
    qw = qv + qc
    direct = qw <= q_test

    # Newton iteration, unrolled MAXITER=10 times, then the Fortran's stopping rule
    # applied by *selection* rather than by masking each step.
    #
    # Masking inside the loop is the obvious encoding and is ruinously expensive here:
    # `twork_i = where(active, step(twork_{i-1}), twork_{i-1})` names twork_{i-1} four
    # times instead of three, and over ten unrolled steps that is 4^10/3^10 ~ 18x the
    # expression tree. Measured, it took gtfn_cpu from 25 s to over 10 minutes for this
    # one stencil.
    #
    # The selection form below is exact and cheap. The Fortran stops at the first step
    # whose change is <= tol, and up to that point its iterates are identical to the
    # unconditional ones -- masking never altered anything before the stop. So computing
    # the unconditional chain t1..t10 and then picking the one the Fortran would have
    # stopped at gives the same double, while each t_i is built once and referenced once
    # more by the cascade. Measured on gtfn_cpu: 25 s unconditional (inexact), 58 s this
    # way (exact), >600 s masked (exact). Exactness for 2.3x, rather than for 24x.
    t0 = temperature
    t1 = _newton_step(t0, temperature, qv, rho, lwdocvd)
    t2 = _newton_step(t1, temperature, qv, rho, lwdocvd)
    t3 = _newton_step(t2, temperature, qv, rho, lwdocvd)
    t4 = _newton_step(t3, temperature, qv, rho, lwdocvd)
    t5 = _newton_step(t4, temperature, qv, rho, lwdocvd)
    t6 = _newton_step(t5, temperature, qv, rho, lwdocvd)
    t7 = _newton_step(t6, temperature, qv, rho, lwdocvd)
    t8 = _newton_step(t7, temperature, qv, rho, lwdocvd)
    t9 = _newton_step(t8, temperature, qv, rho, lwdocvd)
    t10 = _newton_step(t9, temperature, qv, rho, lwdocvd)

    # `go_i` is "the Fortran had not stopped before step i", i.e. every earlier step
    # moved twork by more than tol. Step 1 always runs, matching its
    # `tworkold = twork + 10.0` priming.
    tol = _SatadConst.TOL
    go2 = abs(t1 - t0) > tol
    go3 = go2 & (abs(t2 - t1) > tol)
    go4 = go3 & (abs(t3 - t2) > tol)
    go5 = go4 & (abs(t4 - t3) > tol)
    go6 = go5 & (abs(t5 - t4) > tol)
    go7 = go6 & (abs(t6 - t5) > tol)
    go8 = go7 & (abs(t7 - t6) > tol)
    go9 = go8 & (abs(t8 - t7) > tol)
    go10 = go9 & (abs(t9 - t8) > tol)

    twork = where(
        go10, t10,
        where(go9, t9,
        where(go8, t8,
        where(go7, t7,
        where(go6, t6,
        where(go5, t5,
        where(go4, t4,
        where(go3, t3,
        where(go2, t2, t1))))))))
    )  # fmt: skip

    qwa = _qsat_rho(twork, rho)
    qc_newton = maximum(qc + qv - qwa, _SatadConst.Q_MIN)
    qv_newton = qwa

    new_temperature = where(direct, t_test, twork)
    new_qv = where(direct, qw, qv_newton)
    new_qc = where(direct, 0.0, qc_newton)
    return new_temperature, new_qv, new_qc


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def satad_program(
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    out_temperature: fa.CellKField[ta.wpfloat],
    out_qv: fa.CellKField[ta.wpfloat],
    out_qc: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    satad(
        temperature,
        qv,
        qc,
        rho,
        out=(out_temperature, out_qv, out_qc),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

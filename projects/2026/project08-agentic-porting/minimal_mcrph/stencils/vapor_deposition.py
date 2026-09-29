"""vapor_dep_relaxation: depositional growth/sublimation of ice, snow, graupel, hail.

Mirrors mo_2mom_mcrph_processes.f90::vapor_dep_relaxation (lines 1319-1463) and
::vapor_deposition_generic (lines 1466-1500).

**Simplification, not an approximation, in vapor_deposition_generic's
ventilation term:** the Fortran recomputes
`vent_coeff_b(prtcl,1) * N_sc**n_f / sqrt(nu_l)` from scratch inside the loop
(line 1493) instead of reusing the precomputed `coeffs%b_f` -- but that
recomputed expression is *exactly* `setup_particle_coeffs`'s definition of
`b_f` (mo_2mom_mcrph_processes.f90:1509: `b_f = vent_coeff_b(ptype,1) *
N_sc**n_f / sqrt(nu_l)`), word for word. A commented-out line directly above
it in the source (`!f_v = ( coeffs%a_f + coeffs%b_f * SQRT(D*v) ) * 2.0_wp`)
confirms this was always meant to be `coeffs%b_f`. Using the precomputed
`b_f` here avoids needing a runtime gamma-function evaluation inside a
per-level stencil (which GT4Py has no builtin for anyway) and is
mathematically identical, not an approximation.
"""

import enum

import gt4py.next as gtx
from gt4py.next import abs, exp, log, maximum, minimum, sqrt, where  # noqa: A004

from icon4py.model.common import dimension as dims
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common import type_alias as ta

from minimal_mcrph.stencils.particle_helpers import particle_meanmass
from minimal_mcrph.stencils.saturation import diffusivity, e_es


class _VaporDepositionConst(ta.wpfloat, enum.Enum):
    """See unit_conversion.py's ``_UnitConversionConst`` docstring for why
    these must be enum members rather than plain module floats."""

    PI = 3.14159265358979323846
    T_3 = 273.15
    R_D_VAPOR = 461.51  # water vapor gas constant, "R_d" alias in the Fortran
    K_T = 2.40e-2  # con0_h, thermal conductivity of dry air
    L_ED = 2.8345e6  # als, latent heat of sublimation
    DEP_N_FAC = 0.5  # reduce_sublimation's number-reduction tuning factor
    EPS = 1.0e-20


@gtx.field_operator
def _saturation_terms(
    temperature: fa.CellKField[ta.wpfloat],
    pressure: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    """g_i, s_si. mo_2mom_mcrph_processes.f90:1359-1372."""
    active = temperature < _VaporDepositionConst.T_3
    e_si = e_es(temperature)
    e_d = qv * _VaporDepositionConst.R_D_VAPOR * temperature
    s_si = e_d / e_si - 1.0
    d_vtp = diffusivity(temperature, pressure)
    g_i = (4.0 * _VaporDepositionConst.PI) / (
        _VaporDepositionConst.L_ED
        * _VaporDepositionConst.L_ED
        / (_VaporDepositionConst.K_T * _VaporDepositionConst.R_D_VAPOR * temperature * temperature)
        + _VaporDepositionConst.R_D_VAPOR * temperature / (d_vtp * e_si)
    )
    return where(active, g_i, 0.0), where(active, s_si, 0.0)


@gtx.field_operator
def _raw_deposition_rate(
    x_min: ta.wpfloat,
    x_max: ta.wpfloat,
    a_geo: ta.wpfloat,
    b_geo: ta.wpfloat,
    a_vel: ta.wpfloat,
    b_vel: ta.wpfloat,
    a_ven: ta.wpfloat,
    a_f: ta.wpfloat,
    b_f: ta.wpfloat,
    c_i: ta.wpfloat,
    q: fa.CellKField[ta.wpfloat],
    n: fa.CellKField[ta.wpfloat],
    rho_v: fa.CellKField[ta.wpfloat],
    g_i: fa.CellKField[ta.wpfloat],
    s_si: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    """vapor_deposition_generic, mo_2mom_mcrph_processes.f90:1466-1500."""
    x = particle_meanmass(q, n, x_min, x_max)
    diameter = a_geo * exp(b_geo * log(x))
    velocity = a_vel * exp(b_vel * log(x)) * rho_v
    f_v = maximum(a_f + b_f * sqrt(diameter * velocity), a_f / a_ven)
    dep_q = g_i * n * c_i * diameter * f_v * s_si * dt
    return where(q == 0.0, 0.0, dep_q)


@gtx.field_operator
def vapor_dep_relaxation(
    dt: ta.wpfloat,
    ice_x_min: ta.wpfloat,
    ice_x_max: ta.wpfloat,
    ice_a_geo: ta.wpfloat,
    ice_b_geo: ta.wpfloat,
    ice_a_vel: ta.wpfloat,
    ice_b_vel: ta.wpfloat,
    ice_a_ven: ta.wpfloat,
    ice_a_f: ta.wpfloat,
    ice_b_f: ta.wpfloat,
    ice_c_i: ta.wpfloat,
    snow_x_min: ta.wpfloat,
    snow_x_max: ta.wpfloat,
    snow_a_geo: ta.wpfloat,
    snow_b_geo: ta.wpfloat,
    snow_a_vel: ta.wpfloat,
    snow_b_vel: ta.wpfloat,
    snow_a_ven: ta.wpfloat,
    snow_a_f: ta.wpfloat,
    snow_b_f: ta.wpfloat,
    snow_c_i: ta.wpfloat,
    graupel_x_min: ta.wpfloat,
    graupel_x_max: ta.wpfloat,
    graupel_a_geo: ta.wpfloat,
    graupel_b_geo: ta.wpfloat,
    graupel_a_vel: ta.wpfloat,
    graupel_b_vel: ta.wpfloat,
    graupel_a_ven: ta.wpfloat,
    graupel_a_f: ta.wpfloat,
    graupel_b_f: ta.wpfloat,
    graupel_c_i: ta.wpfloat,
    hail_x_min: ta.wpfloat,
    hail_x_max: ta.wpfloat,
    hail_a_geo: ta.wpfloat,
    hail_b_geo: ta.wpfloat,
    hail_a_vel: ta.wpfloat,
    hail_b_vel: ta.wpfloat,
    hail_a_ven: ta.wpfloat,
    hail_a_f: ta.wpfloat,
    hail_b_f: ta.wpfloat,
    hail_c_i: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    pressure: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    ice_q: fa.CellKField[ta.wpfloat],
    ice_n: fa.CellKField[ta.wpfloat],
    ice_rho_v: fa.CellKField[ta.wpfloat],
    snow_q: fa.CellKField[ta.wpfloat],
    snow_n: fa.CellKField[ta.wpfloat],
    snow_rho_v: fa.CellKField[ta.wpfloat],
    graupel_q: fa.CellKField[ta.wpfloat],
    graupel_n: fa.CellKField[ta.wpfloat],
    graupel_rho_v: fa.CellKField[ta.wpfloat],
    hail_q: fa.CellKField[ta.wpfloat],
    hail_n: fa.CellKField[ta.wpfloat],
    hail_rho_v: fa.CellKField[ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    g_i, s_si = _saturation_terms(temperature, pressure, qv)

    dep_ice = _raw_deposition_rate(
        ice_x_min, ice_x_max, ice_a_geo, ice_b_geo, ice_a_vel, ice_b_vel, ice_a_ven,
        ice_a_f, ice_b_f, ice_c_i, ice_q, ice_n, ice_rho_v, g_i, s_si, dt,
    )  # fmt: skip
    dep_snow = _raw_deposition_rate(
        snow_x_min, snow_x_max, snow_a_geo, snow_b_geo, snow_a_vel, snow_b_vel, snow_a_ven,
        snow_a_f, snow_b_f, snow_c_i, snow_q, snow_n, snow_rho_v, g_i, s_si, dt,
    )  # fmt: skip
    dep_graupel = _raw_deposition_rate(
        graupel_x_min, graupel_x_max, graupel_a_geo, graupel_b_geo, graupel_a_vel, graupel_b_vel,
        graupel_a_ven, graupel_a_f, graupel_b_f, graupel_c_i, graupel_q, graupel_n, graupel_rho_v,
        g_i, s_si, dt,
    )  # fmt: skip
    dep_hail = _raw_deposition_rate(
        hail_x_min, hail_x_max, hail_a_geo, hail_b_geo, hail_a_vel, hail_b_vel, hail_a_ven,
        hail_a_f, hail_b_f, hail_c_i, hail_q, hail_n, hail_rho_v, g_i, s_si, dt,
    )  # fmt: skip

    active = temperature < _VaporDepositionConst.T_3
    qvsidiff = qv - e_es(temperature) / (_VaporDepositionConst.R_D_VAPOR * temperature)
    relax_active = active & (abs(qvsidiff) > _VaporDepositionConst.EPS)

    zdt = 1.0 / dt
    tau_i = zdt / qvsidiff * dep_ice
    tau_s = zdt / qvsidiff * dep_snow
    tau_g = zdt / qvsidiff * dep_graupel
    tau_h = zdt / qvsidiff * dep_hail
    xi_i = tau_i + tau_s + tau_g + tau_h

    xfac = where(
        xi_i < _VaporDepositionConst.EPS, 0.0, qvsidiff / xi_i * (1.0 - exp(-dt * xi_i))
    )

    dep_ice_r = xfac * tau_i
    dep_snow_r = xfac * tau_s
    dep_graupel_r = xfac * tau_g
    dep_hail_r = xfac * tau_h

    negative = qvsidiff < 0.0
    dep_ice_r = where(negative, maximum(dep_ice_r, -ice_q), dep_ice_r)
    dep_snow_r = where(negative, maximum(dep_snow_r, -snow_q), dep_snow_r)
    dep_graupel_r = where(negative, maximum(dep_graupel_r, -graupel_q), dep_graupel_r)
    dep_hail_r = where(negative, maximum(dep_hail_r, -hail_q), dep_hail_r)

    dep_sum = dep_ice_r + dep_graupel_r + dep_snow_r + dep_hail_r

    x_i = particle_meanmass(ice_q, ice_n, ice_x_min, ice_x_max)
    x_s = particle_meanmass(snow_q, snow_n, snow_x_min, snow_x_max)
    x_g = particle_meanmass(graupel_q, graupel_n, graupel_x_min, graupel_x_max)
    x_h = particle_meanmass(hail_q, hail_n, hail_x_min, hail_x_max)

    dep_ice_n = minimum(dep_ice_r, 0.0) / x_i
    dep_snow_n = minimum(dep_snow_r, 0.0) / x_s
    dep_graupel_n = minimum(dep_graupel_r, 0.0) / x_g
    dep_hail_n = minimum(dep_hail_r, 0.0) / x_h

    new_ice_q = where(relax_active, ice_q + dep_ice_r, ice_q)
    new_snow_q = where(relax_active, snow_q + dep_snow_r, snow_q)
    new_graupel_q = where(relax_active, graupel_q + dep_graupel_r, graupel_q)
    new_hail_q = where(relax_active, hail_q + dep_hail_r, hail_q)
    new_qv = where(relax_active, qv - dep_sum, qv)

    new_ice_n = where(
        relax_active,
        maximum(ice_n + _VaporDepositionConst.DEP_N_FAC * dep_ice_n, 0.0),
        ice_n,
    )
    new_snow_n = where(
        relax_active,
        maximum(snow_n + _VaporDepositionConst.DEP_N_FAC * dep_snow_n, 0.0),
        snow_n,
    )
    new_graupel_n = where(
        relax_active,
        maximum(graupel_n + _VaporDepositionConst.DEP_N_FAC * dep_graupel_n, 0.0),
        graupel_n,
    )
    new_hail_n = where(
        relax_active,
        maximum(hail_n + _VaporDepositionConst.DEP_N_FAC * dep_hail_n, 0.0),
        hail_n,
    )

    return (
        new_ice_q,
        new_ice_n,
        new_snow_q,
        new_snow_n,
        new_graupel_q,
        new_graupel_n,
        new_hail_q,
        new_hail_n,
        new_qv,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def vapor_dep_relaxation_program(
    dt: ta.wpfloat,
    ice_x_min: ta.wpfloat,
    ice_x_max: ta.wpfloat,
    ice_a_geo: ta.wpfloat,
    ice_b_geo: ta.wpfloat,
    ice_a_vel: ta.wpfloat,
    ice_b_vel: ta.wpfloat,
    ice_a_ven: ta.wpfloat,
    ice_a_f: ta.wpfloat,
    ice_b_f: ta.wpfloat,
    ice_c_i: ta.wpfloat,
    snow_x_min: ta.wpfloat,
    snow_x_max: ta.wpfloat,
    snow_a_geo: ta.wpfloat,
    snow_b_geo: ta.wpfloat,
    snow_a_vel: ta.wpfloat,
    snow_b_vel: ta.wpfloat,
    snow_a_ven: ta.wpfloat,
    snow_a_f: ta.wpfloat,
    snow_b_f: ta.wpfloat,
    snow_c_i: ta.wpfloat,
    graupel_x_min: ta.wpfloat,
    graupel_x_max: ta.wpfloat,
    graupel_a_geo: ta.wpfloat,
    graupel_b_geo: ta.wpfloat,
    graupel_a_vel: ta.wpfloat,
    graupel_b_vel: ta.wpfloat,
    graupel_a_ven: ta.wpfloat,
    graupel_a_f: ta.wpfloat,
    graupel_b_f: ta.wpfloat,
    graupel_c_i: ta.wpfloat,
    hail_x_min: ta.wpfloat,
    hail_x_max: ta.wpfloat,
    hail_a_geo: ta.wpfloat,
    hail_b_geo: ta.wpfloat,
    hail_a_vel: ta.wpfloat,
    hail_b_vel: ta.wpfloat,
    hail_a_ven: ta.wpfloat,
    hail_a_f: ta.wpfloat,
    hail_b_f: ta.wpfloat,
    hail_c_i: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    pressure: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    ice_q: fa.CellKField[ta.wpfloat],
    ice_n: fa.CellKField[ta.wpfloat],
    ice_rho_v: fa.CellKField[ta.wpfloat],
    snow_q: fa.CellKField[ta.wpfloat],
    snow_n: fa.CellKField[ta.wpfloat],
    snow_rho_v: fa.CellKField[ta.wpfloat],
    graupel_q: fa.CellKField[ta.wpfloat],
    graupel_n: fa.CellKField[ta.wpfloat],
    graupel_rho_v: fa.CellKField[ta.wpfloat],
    hail_q: fa.CellKField[ta.wpfloat],
    hail_n: fa.CellKField[ta.wpfloat],
    hail_rho_v: fa.CellKField[ta.wpfloat],
    out_ice_q: fa.CellKField[ta.wpfloat],
    out_ice_n: fa.CellKField[ta.wpfloat],
    out_snow_q: fa.CellKField[ta.wpfloat],
    out_snow_n: fa.CellKField[ta.wpfloat],
    out_graupel_q: fa.CellKField[ta.wpfloat],
    out_graupel_n: fa.CellKField[ta.wpfloat],
    out_hail_q: fa.CellKField[ta.wpfloat],
    out_hail_n: fa.CellKField[ta.wpfloat],
    out_qv: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    vapor_dep_relaxation(
        dt,
        ice_x_min, ice_x_max, ice_a_geo, ice_b_geo, ice_a_vel, ice_b_vel, ice_a_ven,
        ice_a_f, ice_b_f, ice_c_i,
        snow_x_min, snow_x_max, snow_a_geo, snow_b_geo, snow_a_vel, snow_b_vel, snow_a_ven,
        snow_a_f, snow_b_f, snow_c_i,
        graupel_x_min, graupel_x_max, graupel_a_geo, graupel_b_geo, graupel_a_vel, graupel_b_vel,
        graupel_a_ven, graupel_a_f, graupel_b_f, graupel_c_i,
        hail_x_min, hail_x_max, hail_a_geo, hail_b_geo, hail_a_vel, hail_b_vel, hail_a_ven,
        hail_a_f, hail_b_f, hail_c_i,
        temperature, pressure, qv,
        ice_q, ice_n, ice_rho_v,
        snow_q, snow_n, snow_rho_v,
        graupel_q, graupel_n, graupel_rho_v,
        hail_q, hail_n, hail_rho_v,
        out=(
            out_ice_q, out_ice_n,
            out_snow_q, out_snow_n,
            out_graupel_q, out_graupel_n,
            out_hail_q, out_hail_n,
            out_qv,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )  # fmt: skip

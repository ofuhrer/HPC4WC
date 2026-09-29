# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import NamedTuple

import gt4py.next as gtx
from gt4py.next import broadcast, maximum, minimum, power, sqrt, where
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import (
    GraupelConsts,
    IndexConsts,
    ThermodynamicConsts,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q, Q_scalar
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import (
    _deposition_auto_conversion,
    _deposition_factor,
    _ice_deposition_nucleation,
    _ice_mass,
    _ice_number,
    _ice_sticking,
    _snow_lambda,
    _snow_number,
    _vel_scale_factor_default_scalar,
    _vel_scale_factor_ice_scalar,
    _vel_scale_factor_snow_scalar,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import (
    _internal_energy_scalar,
    _qsat_ice_rho,
    _qsat_rho,
    _qsat_rho_tmelt,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import (
    _cloud_to_graupel,
    _cloud_to_rain,
    _cloud_to_snow,
    _cloud_x_ice,
    _graupel_to_rain,
    _ice_to_graupel,
    _ice_to_snow,
    _rain_to_graupel,
    _rain_to_vapor,
    _snow_to_graupel,
    _snow_to_rain,
    _vapor_x_graupel,
    _vapor_x_ice,
    _vapor_x_snow,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import wpfloat


class PrecipStateQx(NamedTuple):
    x: ta.wpfloat
    p: ta.wpfloat
    vc: ta.wpfloat
    activated: bool


class TempState(NamedTuple):
    t: ta.wpfloat
    eflx: ta.wpfloat
    activated: bool


class IntegrationState(NamedTuple):
    r: PrecipStateQx
    s: PrecipStateQx
    i: PrecipStateQx
    g: PrecipStateQx
    t_state: TempState
    rho: ta.wpfloat
    pflx_tot: ta.wpfloat


@gtx.field_operator
def precip_qx_level_update(  # noqa: PLR0917 [too-many-positional-arguments]
    previous_level_q: PrecipStateQx,
    previous_level_rho: ta.wpfloat,
    prefactor: ta.wpfloat,  # param[0] of fall_speed
    exponent: ta.wpfloat,  # param[1] of fall_speed
    offset: ta.wpfloat,  # param[1] of fall_speed
    zeta: ta.wpfloat,  # dt/(2dz)
    vc: ta.wpfloat,  # state dependent fall speed correction
    q: ta.wpfloat,  # specific mass of hydrometeor
    rho: ta.wpfloat,  # density
    mask: bool,
) -> PrecipStateQx:
    current_level_activated = previous_level_q.activated | mask
    rho_x = q * rho
    flx_eff = (rho_x / zeta) + wpfloat(2.0) * previous_level_q.p
    # Inlined calculation using _fall_speed_scalar
    flx_partial = minimum(rho_x * vc * prefactor * power((rho_x + offset), exponent), flx_eff)

    rhox_prev = (previous_level_q.x + q) * wpfloat(0.5) * previous_level_rho

    if current_level_activated:
        vt = (
            previous_level_q.vc * prefactor * power((rhox_prev + offset), exponent)
            if previous_level_q.activated
            else wpfloat(0.0)
        )
        x = (zeta * (flx_eff - flx_partial)) / ((wpfloat(1.0) + zeta * vt) * rho)  # q update
        p = (x * rho * vt + flx_partial) * wpfloat(0.5)  # flux
    else:
        x = q
        p = wpfloat(0.0)

    return PrecipStateQx(
        x=x,
        p=p,
        vc=vc,
        activated=current_level_activated,
    )


@gtx.field_operator
def _temperature_update(  # noqa: PLR0917 [too-many-positional-arguments]
    previous_level: TempState,
    t: ta.wpfloat,
    t_kp1: ta.wpfloat,
    pr: ta.wpfloat,  # precipitable rain
    pflx_tot: ta.wpfloat,  # total precipitation flux
    q: Q_scalar,
    qliq: ta.wpfloat,
    qice: ta.wpfloat,
    rho: ta.wpfloat,  # density
    dz: ta.wpfloat,
    dt: ta.wpfloat,
    mask: bool,
) -> TempState:
    current_level_activated = previous_level.activated | mask
    if current_level_activated:
        eflx = pr * (
            ThermodynamicConsts.clw * t - ThermodynamicConsts.cvd * t_kp1 - GraupelConsts.lvc
        ) + (pflx_tot) * (
            GraupelConsts.ci * t - ThermodynamicConsts.cvd * t_kp1 - GraupelConsts.lsc
        )

        e_int = (
            _internal_energy_scalar(
                t=t, qv=q.v, qliq=q.c + q.r, qice=q.s + q.i + q.g, rho=rho, dz=dz
            )
            + dt * previous_level.eflx
            - dt * eflx
        )

        #  Inlined calculation using T_from_internal_energy_scalar
        #  in order to avoid scan_operator -> field_operator
        qtot = qliq + qice + q.v  # total water specific mass
        cv = (
            (
                ThermodynamicConsts.cvd * (wpfloat(1.0) - qtot)
                + ThermodynamicConsts.cvv * q.v
                + ThermodynamicConsts.clw * qliq
                + GraupelConsts.ci * qice
            )
            * rho
            * dz
        )  # Moist isometric specific heat
        t = (e_int + rho * dz * (qliq * GraupelConsts.lvc + qice * GraupelConsts.lsc)) / cv
    else:
        eflx = previous_level.eflx

    return TempState(t=t, eflx=eflx, activated=current_level_activated)


@gtx.scan_operator(
    axis=dims.KDim,
    forward=True,
    init=IntegrationState(
        r=PrecipStateQx(x=0.0, p=0.0, vc=0.0, activated=False),
        s=PrecipStateQx(x=0.0, p=0.0, vc=0.0, activated=False),
        i=PrecipStateQx(x=0.0, p=0.0, vc=0.0, activated=False),
        g=PrecipStateQx(x=0.0, p=0.0, vc=0.0, activated=False),
        t_state=TempState(t=0.0, eflx=0.0, activated=False),
        rho=0.0,
        pflx_tot=0.0,
    ),
)
def _precip_and_t(  # noqa: PLR0917 [too-many-positional-arguments]
    previous_level: IntegrationState,
    t: ta.wpfloat,
    t_kp1: ta.wpfloat,
    rho: ta.wpfloat,  # density
    q: Q_scalar,
    mask_r: bool,
    mask_s: bool,
    mask_i: bool,
    mask_g: bool,
    dt: ta.wpfloat,
    dz: ta.wpfloat,
) -> IntegrationState:
    zeta = dt / (wpfloat(2.0) * dz)
    xrho = sqrt(GraupelConsts.rho_00 / rho)

    vc_r = _vel_scale_factor_default_scalar(xrho)
    vc_s = _vel_scale_factor_snow_scalar(xrho, rho, t, q.s)
    vc_i = _vel_scale_factor_ice_scalar(xrho)
    vc_g = _vel_scale_factor_default_scalar(xrho)
    any_mask = mask_r | mask_s | mask_i | mask_g
    previous_level_activated = (
        previous_level.r.activated
        | previous_level.s.activated
        | previous_level.i.activated
        | previous_level.g.activated
        | previous_level.t_state.activated
    )
    current_level_activated = any_mask | previous_level_activated
    # TODO(): Use of combined if-statement to reduce checks in case any of the masks or previous levels are not activated. Can be made unnecessary with future transformations.
    if current_level_activated:
        r_update = precip_qx_level_update(
            previous_level.r,
            previous_level.rho,
            IndexConsts.prefactor_r,
            IndexConsts.exponent_r,
            IndexConsts.offset_r,
            zeta,
            vc_r,
            q.r,
            rho,
            mask_r,
        )
        s_update = precip_qx_level_update(
            previous_level.s,
            previous_level.rho,
            IndexConsts.prefactor_s,
            IndexConsts.exponent_s,
            IndexConsts.offset_s,
            zeta,
            vc_s,
            q.s,
            rho,
            mask_s,
        )
        i_update = precip_qx_level_update(
            previous_level.i,
            previous_level.rho,
            IndexConsts.prefactor_i,
            IndexConsts.exponent_i,
            IndexConsts.offset_i,
            zeta,
            vc_i,
            q.i,
            rho,
            mask_i,
        )
        g_update = precip_qx_level_update(
            previous_level.g,
            previous_level.rho,
            IndexConsts.prefactor_g,
            IndexConsts.exponent_g,
            IndexConsts.offset_g,
            zeta,
            vc_g,
            q.g,
            rho,
            mask_g,
        )

        qliq = q.c + r_update.x
        qice = s_update.x + i_update.x + g_update.x
        t_update = _temperature_update(
            previous_level.t_state,
            t=t,
            t_kp1=t_kp1,
            pr=r_update.p,
            pflx_tot=s_update.p + i_update.p + g_update.p,
            q=q,
            qliq=qliq,
            qice=qice,
            rho=rho,
            dz=dz,
            dt=dt,
            mask=any_mask,
        )
    else:
        r_update = PrecipStateQx(x=q.r, p=wpfloat(0.0), vc=vc_r, activated=False)
        s_update = PrecipStateQx(x=q.s, p=wpfloat(0.0), vc=vc_s, activated=False)
        i_update = PrecipStateQx(x=q.i, p=wpfloat(0.0), vc=vc_i, activated=False)
        g_update = PrecipStateQx(x=q.g, p=wpfloat(0.0), vc=vc_g, activated=False)
        t_update = TempState(t=t, eflx=previous_level.t_state.eflx, activated=False)

    return IntegrationState(
        r=r_update,
        s=s_update,
        i=i_update,
        g=g_update,
        t_state=t_update,
        rho=rho,
        pflx_tot=s_update.p + i_update.p + g_update.p + r_update.p,
    )


@gtx.field_operator
def symmetric(
    transition: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    return maximum(transition, wpfloat(0.0)), -minimum(transition, wpfloat(0.0))


@gtx.field_operator
def cond_symmetric(
    condition: fa.CellKField[bool],
    transition: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    return where(condition, symmetric(transition), (wpfloat(0.0), wpfloat(0.0)))


# TODO(havogt): this is an example for expandable parameters + reduce over the expandable parameters
@gtx.field_operator
def sink_saturation(
    t: tuple[
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
    ],
    x: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
    where_: fa.CellKField[bool],
):
    sink = where(where_, t[0] + t[1] + t[2] + t[3], wpfloat(0.0))
    stot = x / dt
    sink_saturated = (sink > stot) & (x > GraupelConsts.qmin)
    t0 = where(sink_saturated, t[0] * stot / sink, t[0])
    t1 = where(sink_saturated, t[1] * stot / sink, t[1])
    t2 = where(sink_saturated, t[2] * stot / sink, t[2])
    t3 = where(sink_saturated, t[3] * stot / sink, t[3])
    sink = where(sink_saturated, t0 + t1 + t2 + t3, sink)
    return sink, t0, t1, t2, t3


@gtx.field_operator
def _q_t_update(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],
    p: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    q: Q,
    dt: ta.wpfloat,
    qnc: ta.wpfloat,
    enable_masking: bool,
) -> tuple[
    Q,
    fa.CellKField[ta.wpfloat],
]:
    if enable_masking:
        is_sig_present = maximum(q.g, maximum(q.i, q.s)) > GraupelConsts.qmin
    else:
        is_sig_present = broadcast(True, (dims.CellDim, dims.KDim))

    dvsw = q.v - _qsat_rho(t, rho)
    qvsi = _qsat_ice_rho(t, rho)
    dvsi = q.v - qvsi
    n_snow = _snow_number(t, rho, q.s)

    l_snow = _snow_lambda(rho, q.s, n_snow)

    t_below_tmelt = t < ThermodynamicConsts.tmelt
    t_at_least_tmelt = ~t_below_tmelt

    # Define conversion 'matrix'
    c2r = _cloud_to_rain(t, q.c, q.r, qnc)
    r2v = _rain_to_vapor(t, rho, q.c, q.r, dvsw, dt)
    c2i, i2c = symmetric(_cloud_x_ice(t, q.c, q.i, dt))

    c2s = _cloud_to_snow(t, q.c, q.s, n_snow, l_snow)
    c2g = _cloud_to_graupel(t, rho, q.c, q.g)

    c2r = where(t_at_least_tmelt, c2r + c2s + c2g, c2r)
    c2s = where(t_at_least_tmelt, wpfloat(0.0), c2s)
    c2g = where(t_at_least_tmelt, wpfloat(0.0), c2g)

    n_ice = _ice_number(t, rho)
    m_ice = _ice_mass(q.i, n_ice)
    x_ice = _ice_sticking(t)

    eta = where(t_below_tmelt, _deposition_factor(t, qvsi), wpfloat(0.0))
    v2i, i2v = cond_symmetric(
        t_below_tmelt & is_sig_present, _vapor_x_ice(q.i, m_ice, eta, dvsi, rho, dt)
    )
    v2i = where(
        t_below_tmelt, v2i + _ice_deposition_nucleation(t, q.c, q.i, n_ice, dvsi, dt), wpfloat(0.0)
    )  # 0.0 or v2i both OK

    ice_dep = where(t_below_tmelt, minimum(v2i, dvsi / dt), wpfloat(0.0))
    # TODO(): _deposition_auto_conversion yields roundoff differences in i2s
    i2s = where(
        t_below_tmelt & is_sig_present,
        _deposition_auto_conversion(q.i, m_ice, ice_dep) + _ice_to_snow(q.i, n_snow, l_snow, x_ice),
        wpfloat(0.0),
    )
    i2g = where(
        t_below_tmelt & is_sig_present, _ice_to_graupel(rho, q.r, q.g, q.i, x_ice), wpfloat(0.0)
    )
    s2g = where(t_below_tmelt & is_sig_present, _snow_to_graupel(t, rho, q.c, q.s), wpfloat(0.0))
    r2g = where(
        t_below_tmelt & is_sig_present,
        _rain_to_graupel(t, rho, q.c, q.r, q.i, q.s, m_ice, dvsw, dt),
        wpfloat(0.0),
    )

    dvsw0 = q.v - _qsat_rho_tmelt(rho)
    v2s, s2v = cond_symmetric(
        is_sig_present,
        _vapor_x_snow(t, p, rho, q.s, n_snow, l_snow, eta, ice_dep, dvsw, dvsi, dvsw0, dt),
    )

    v2g, g2v = cond_symmetric(
        is_sig_present, _vapor_x_graupel(t, p, rho, q.g, dvsw, dvsi, dvsw0, dt)
    )

    s2r = where(is_sig_present, _snow_to_rain(t, p, rho, dvsw0, q.s), wpfloat(0.0))
    g2r = where(is_sig_present, _graupel_to_rain(t, p, rho, dvsw0, q.g), wpfloat(0.0))

    # The following transitions are not physically meaningful, would be 0.0 in other implementation
    # here they are simply never used:
    # identity transitions v_v, c_c, ... g_g
    # unphysical transitions: v_c, v_r, c_v, r_c, r_s, r_i, s_c, s_i, i_r, g_c, g_s, g_i
    # Physical: v_s, v_i, v_g, c_r, c_s, c_i, c_g, r_v, r_g, s_v, s_r, s_g, i_v, i_c, i_s, i_g, g_v, g_r
    # SINK calculation

    UNUSED = broadcast(wpfloat(0.0), (dims.CellDim, dims.KDim))
    EVERYWHERE = broadcast(True, (dims.CellDim, dims.KDim))
    sink_v, v2s, v2i, v2g, _ = sink_saturation((v2s, v2i, v2g, UNUSED), q.v, dt, where_=EVERYWHERE)
    sink_c, c2r, c2s, c2i, c2g = sink_saturation((c2r, c2s, c2i, c2g), q.c, dt, where_=EVERYWHERE)
    sink_r, r2v, r2g, _, __ = sink_saturation(
        (r2v, r2g, UNUSED, UNUSED), q.r, dt, where_=EVERYWHERE
    )
    sink_s, s2v, s2r, s2g, _ = sink_saturation(
        (s2v, s2r, s2g, UNUSED), q.s, dt, where_=is_sig_present
    )
    sink_i, i2v, i2c, i2s, i2g = sink_saturation(
        (i2v, i2c, i2s, i2g), q.i, dt, where_=is_sig_present
    )
    sink_g, g2v, g2r, _, __ = sink_saturation(
        (g2v, g2r, UNUSED, UNUSED), q.g, dt, where_=is_sig_present
    )

    # water content updates:
    # Physical: v_s, v_i, v_g, c_r, c_s, c_i, c_g, r_v, r_g, s_v, s_r, s_g, i_v, i_c, i_s, i_g, g_v, g_r
    dqdt_v = r2v + s2v + i2v + g2v - sink_v  # Missing: c2v
    qv = maximum(wpfloat(0.0), q.v + dqdt_v * dt)
    dqdt_c = i2c - sink_c  # Missing: v2c, r2c, s2c, g2c
    qc = maximum(wpfloat(0.0), q.c + dqdt_c * dt)
    dqdt_r = c2r + s2r + g2r - sink_r  # Missing: v2r + i2r
    qr = maximum(wpfloat(0.0), q.r + dqdt_r * dt)
    dqdt_s = v2s + c2s + i2s - sink_s  # Missing: r2s + g2s
    qs = maximum(wpfloat(0.0), q.s + dqdt_s * dt)
    dqdt_i = v2i + c2i - sink_i  # Missing: r2i + s2i + g2i
    qi = maximum(wpfloat(0.0), q.i + dqdt_i * dt)
    dqdt_g = v2g + c2g + r2g + s2g + i2g - sink_g
    qg = maximum(wpfloat(0.0), q.g + dqdt_g * dt)

    qice = qs + qi + qg
    qliq = qc + qr
    qtot = qv + qice + qliq

    cv = (
        ThermodynamicConsts.cvd
        + (ThermodynamicConsts.cvv - ThermodynamicConsts.cvd) * qtot
        + (ThermodynamicConsts.clw - ThermodynamicConsts.cvv) * qliq
        + (GraupelConsts.ci - ThermodynamicConsts.cvv) * qice
    )
    t = (
        t
        + dt
        * (
            (dqdt_c + dqdt_r)
            * (GraupelConsts.lvc - (ThermodynamicConsts.clw - ThermodynamicConsts.cvv) * t)
            + (dqdt_i + dqdt_s + dqdt_g)
            * (GraupelConsts.lsc - (GraupelConsts.ci - ThermodynamicConsts.cvv) * t)
        )
        / cv
    )
    return Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg), t


@gtx.field_operator
def _precipitation_effects(  # noqa: PLR0917 [too-many-positional-arguments]
    last_lev: gtx.int32,
    kmin_r: fa.CellKField[bool],  # rain minimum level
    kmin_i: fa.CellKField[bool],  # ice minimum level
    kmin_s: fa.CellKField[bool],  # snow minimum level
    kmin_g: fa.CellKField[bool],  # graupel minimum level
    q_in: Q,
    t: fa.CellKField[ta.wpfloat],  # temperature,
    rho: fa.CellKField[ta.wpfloat],  # density
    dz: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
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
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    t_kp1 = concat_where(dims.KDim < last_lev, t(KDim + 1), t)

    precip_state = _precip_and_t(
        t,
        t_kp1,
        rho,
        q_in,
        kmin_r,
        kmin_s,
        kmin_i,
        kmin_g,
        dt,
        dz,
    )
    qr = precip_state.r.x
    pr = precip_state.r.p
    qs = precip_state.s.x
    ps = precip_state.s.p
    qi = precip_state.i.x
    pi = precip_state.i.p
    qg = precip_state.g.x
    pg = precip_state.g.p

    t = precip_state.t_state.t
    eflx = precip_state.t_state.eflx

    pflx_tot = precip_state.pflx_tot

    return qr, qs, qi, qg, t, pflx_tot, pr, ps, pi, pg, eflx


@gtx.field_operator
def graupel(  # noqa: PLR0917 [too-many-positional-arguments]
    last_level: gtx.int32,
    dz: fa.CellKField[ta.wpfloat],
    te: fa.CellKField[ta.wpfloat],  # Temperature
    p: fa.CellKField[ta.wpfloat],  # Pressure
    rho: fa.CellKField[ta.wpfloat],  # Density containing dry air and water constituents
    q: Q,
    dt: ta.wpfloat,
    qnc: ta.wpfloat,
    enable_masking: bool,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    Q,
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    kmin_r = q.r > GraupelConsts.qmin
    kmin_i = q.i > GraupelConsts.qmin
    kmin_s = q.s > GraupelConsts.qmin
    kmin_g = q.g > GraupelConsts.qmin
    mask = (
        (maximum(q.c, maximum(q.g, maximum(q.i, maximum(q.r, q.s)))) > GraupelConsts.qmin)
        | ((te < GraupelConsts.tfrz_het2) & (q.v > _qsat_ice_rho(te, rho)))
        | ~enable_masking
    )
    q, t = where(mask, _q_t_update(te, p, rho, q, dt, qnc, enable_masking=enable_masking), (q, te))
    qr, qs, qi, qg, t, pflx, pr, ps, pi, pg, pre = _precipitation_effects(
        last_level, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt
    )

    return t, Q(v=q.v, c=q.c, r=qr, s=qs, i=qi, g=qg), pflx, pr, ps, pi, pg, pre


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def graupel_run(  # noqa: PLR0917 [too-many-positional-arguments]
    dz: fa.CellKField[ta.wpfloat],
    te: fa.CellKField[ta.wpfloat],  # Temperature
    p: fa.CellKField[ta.wpfloat],  # Pressure
    rho: fa.CellKField[ta.wpfloat],  # Density containing dry air and water constituents
    q_in: Q,
    dt: ta.wpfloat,  # Time step
    qnc: ta.wpfloat,
    q_out: Q,
    t_out: fa.CellKField[ta.wpfloat],  # Revised temperature
    pflx: fa.CellKField[ta.wpfloat],  # Total precipitation flux
    pr: fa.CellKField[ta.wpfloat],  # Precipitation of rain
    ps: fa.CellKField[ta.wpfloat],  # Precipitation of snow
    pi: fa.CellKField[ta.wpfloat],  # Precipitation of ice
    pg: fa.CellKField[ta.wpfloat],  # Precipitation of graupel
    pre: fa.CellKField[ta.wpfloat],  # Precipitation of graupel
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
    enable_masking: bool,
) -> None:
    graupel(
        last_level=vertical_end - 1,
        dz=dz,
        te=te,
        p=p,
        rho=rho,
        q=q_in,
        dt=dt,
        qnc=qnc,
        enable_masking=enable_masking,
        out=(t_out, q_out, pflx, pr, ps, pi, pg, pre),
        domain=(
            # t_out
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
            # q_out
            (
                {
                    dims.CellDim: (horizontal_start, horizontal_end),
                    dims.KDim: (vertical_start, vertical_end),
                },
                {
                    dims.CellDim: (horizontal_start, horizontal_end),
                    dims.KDim: (vertical_start, vertical_end),
                },
                {
                    dims.CellDim: (horizontal_start, horizontal_end),
                    dims.KDim: (vertical_start, vertical_end),
                },
                {
                    dims.CellDim: (horizontal_start, horizontal_end),
                    dims.KDim: (vertical_start, vertical_end),
                },
                {
                    dims.CellDim: (horizontal_start, horizontal_end),
                    dims.KDim: (vertical_start, vertical_end),
                },
                {
                    dims.CellDim: (horizontal_start, horizontal_end),
                    dims.KDim: (vertical_start, vertical_end),
                },
            ),
            # pflx
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
            # pr
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_end - 1, vertical_end),
            },
            # ps
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_end - 1, vertical_end),
            },
            # pi
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_end - 1, vertical_end),
            },
            # pg
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_end - 1, vertical_end),
            },
            # pre
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_end - 1, vertical_end),
            },
        ),
    )

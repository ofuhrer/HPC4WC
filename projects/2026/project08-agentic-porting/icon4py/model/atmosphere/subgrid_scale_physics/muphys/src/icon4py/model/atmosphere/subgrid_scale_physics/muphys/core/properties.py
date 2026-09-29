# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import exp, maximum, minimum, power, where

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import (
    GraupelConsts,
    ThermodynamicConsts,
)
from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _deposition_auto_conversion(
    qi: fa.CellKField[ta.wpfloat],
    m_ice: fa.CellKField[ta.wpfloat],
    ice_dep: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Perform the automatic conversion of deposition

    Args:
        qi:       Ice specific mass
        m_ice:    Ice crystal mass
        ice_dep:  Rate of ice deposition (some to snow)

    Result:       Conversion rate
    """
    M0_S = wpfloat(3.0e-9)  # Initial mass of snow crystals
    B_DEP = wpfloat(0.666666666666666667)  # Exponent
    XCRIT = wpfloat(1.0)  # Critical threshold parameter

    return where(
        qi > GraupelConsts.qmin,
        maximum(wpfloat(0.0), ice_dep) * B_DEP / (power((M0_S / m_ice), B_DEP) - XCRIT),
        wpfloat(0.0),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def deposition_auto_conversion(
    qi: fa.CellKField[ta.wpfloat],  # Ice specific mass
    m_ice: fa.CellKField[ta.wpfloat],  # Ice crystal mass
    ice_dep: fa.CellKField[ta.wpfloat],  # Rate of ice deposition (some to snow)
    conversion_rate: fa.CellKField[ta.wpfloat],  # output
):
    _deposition_auto_conversion(qi=qi, m_ice=m_ice, ice_dep=ice_dep, out=conversion_rate)


@gtx.field_operator
def _deposition_factor(
    t: fa.CellKField[ta.wpfloat],
    qvsi: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the deposition factor

    Args:
        t:       Temperature
        qvsi:    Saturation (ice) specific vapor mass

    Result:      Deposition factor
    """

    KAPPA = wpfloat(2.40e-2)  # Thermal conductivity of dry air
    B = wpfloat(1.94)  # Exponent
    A = ThermodynamicConsts.als * ThermodynamicConsts.als / (KAPPA * ThermodynamicConsts.rv)  # TBD
    CX = wpfloat(2.22e-5) * power(ThermodynamicConsts.tmelt, (-B)) * wpfloat(101325.0)  # TBD

    x = CX / ThermodynamicConsts.rd * power(t, B - wpfloat(1.0))
    return x / (wpfloat("1.0") + A * x * qvsi / (t * t))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def deposition_factor(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    qvsi: fa.CellKField[ta.wpfloat],  # Saturation (ice) specific vapor mass
    deposition_rate: fa.CellKField[ta.wpfloat],  # deposition rate
):
    _deposition_factor(t=t, qvsi=qvsi, out=deposition_rate)


@gtx.field_operator
def _fall_speed_scalar(
    density: ta.wpfloat,
    prefactor: ta.wpfloat,
    offset: ta.wpfloat,
    exponent: ta.wpfloat,
) -> ta.wpfloat:  # Fall speed
    """
    Compute the scalar fall speed (can be used in scan operator)

    Args:
        density:       Density of species
        prefactor:     Multiplicative factor
        offset:        Linear offset to density
        exponent:      Exponent of power function

    Result:            Fall speed
    """
    return prefactor * power((density + offset), exponent)


@gtx.field_operator
def _fall_speed(
    density: fa.CellKField[ta.wpfloat],
    prefactor: ta.wpfloat,
    offset: ta.wpfloat,
    exponent: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:  # Fall speed
    """
    Compute the fall speed

    Args:
        density:       Density of species
        prefactor:     Multiplicative factor
        offset:        Linear offset to density
        exponent:      Exponent of power function

    Result:            Fall speed
    """
    return prefactor * power((density + offset), exponent)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def fall_speed_scalar(
    density: ta.wpfloat,  # Density of species
    prefactor: ta.wpfloat,
    offset: ta.wpfloat,
    exponent: ta.wpfloat,
    speed: ta.wpfloat,  # output
):
    _fall_speed_scalar(
        density=density, prefactor=prefactor, offset=offset, exponent=exponent, out=speed
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def fall_speed(
    density: fa.CellKField[ta.wpfloat],  # Density of species
    prefactor: ta.wpfloat,
    offset: ta.wpfloat,
    exponent: ta.wpfloat,
    speed: fa.CellKField[ta.wpfloat],  # output
):
    _fall_speed(density=density, prefactor=prefactor, offset=offset, exponent=exponent, out=speed)


@gtx.field_operator
def _ice_deposition_nucleation(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    ni: fa.CellKField[ta.wpfloat],
    dvsi: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:  # Rate of vapor deposition for new ice
    """
    Compute the vapor deposition for new ice

    Args:
        t:            Temperature
        qc:           Specific humidity of cloud
        qi:           Specific humidity of ice
        ni:           Ice crystal number
        dvsi:         Vapor excess with respect to ice saturation
        dt:           Time step
    Result:           Rate of vapor deposition for new ice
    """
    return where(
        (qi <= GraupelConsts.qmin)
        & (
            ((t < GraupelConsts.tfrz_het2) & (dvsi > wpfloat(0.0)))
            | ((t <= GraupelConsts.tfrz_het1) & (qc > GraupelConsts.qmin))
        ),
        minimum(GraupelConsts.m0_ice * ni, maximum(wpfloat(0.0), dvsi)) / dt,
        wpfloat(0.0),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_deposition_nucleation(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],  # Temperature
    qc: fa.CellKField[ta.wpfloat],  # Specific humidity of cloud
    qi: fa.CellKField[ta.wpfloat],  # Specific humidity of ice
    ni: fa.CellKField[ta.wpfloat],  # Ice crystal number
    dvsi: fa.CellKField[ta.wpfloat],  # Vapor excess with respect to ice sat
    dt: ta.wpfloat,  # Time step
    vapor_deposition_rate: fa.CellKField[ta.wpfloat],  # output
):
    _ice_deposition_nucleation(
        t=t, qc=qc, qi=qi, ni=ni, dvsi=dvsi, dt=dt, out=vapor_deposition_rate
    )


@gtx.field_operator
def _ice_mass(
    qi: fa.CellKField[ta.wpfloat],
    ni: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the ice mass

    Args:
        qi:         Specific humidity of ice
        ni:         Ice crystal number

    Result:         Ice mass
    """
    MI_MAX = wpfloat(1.0e-9)
    return maximum(GraupelConsts.m0_ice, minimum(qi / ni, MI_MAX))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_mass(
    qi: fa.CellKField[ta.wpfloat],  # Specific humidity of ice
    ni: fa.CellKField[ta.wpfloat],  # Ice crystal number
    mass: fa.CellKField[ta.wpfloat],  # output
):
    _ice_mass(qi=qi, ni=ni, out=mass)


@gtx.field_operator
def _ice_number(
    t: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Calculate the ice number

    Args:
        t:           Ambient temperature
        rho:         Ambient density

    Result:          Ice number
    """
    A_COOP = wpfloat(5.000)  # Parameter in cooper fit
    B_COOP = wpfloat(0.304)  # Parameter in cooper fit
    NIMAX = wpfloat(250.0e3)  # Maximal number of ice crystals
    return minimum(NIMAX, A_COOP * exp(B_COOP * (ThermodynamicConsts.tmelt - t))) / rho


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_number(
    t: fa.CellKField[ta.wpfloat],  # Ambient temperature
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    number: fa.CellKField[ta.wpfloat],  # output
):
    _ice_number(t=t, rho=rho, out=number)


@gtx.field_operator
def _ice_sticking(
    t: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the ice sticking

    Args:
        t:            Temperature

    Result:           Ice sticking
    """
    A_FREEZ = wpfloat(0.09)  # Scale factor for freezing depression
    B_MAX_EXP = wpfloat(1.00)  # Maximum for exponential temperature factor
    EFF_MIN = wpfloat(0.075)  # Minimum sticking efficiency
    EFF_FAC = wpfloat(3.5e-3)  # Scaling factor [1/K] for cloud ice sticking efficiency
    TCRIT = ThermodynamicConsts.tmelt - wpfloat(
        85.0
    )  # Temperature at which cloud ice autoconversion starts

    return maximum(
        maximum(minimum(exp(A_FREEZ * (t - ThermodynamicConsts.tmelt)), B_MAX_EXP), EFF_MIN),
        EFF_FAC * (t - TCRIT),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_sticking(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    sticking_factor: fa.CellKField[ta.wpfloat],  # output
):
    _ice_sticking(t=t, out=sticking_factor)


@gtx.field_operator
def _snow_lambda(
    rho: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    ns: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the riming snow rate

    Args:
        rho:          Ambient density
        qs:           Snow specific mass
        ns:           Snow number

    Result:           Riming snow rate
    """
    A2 = GraupelConsts.ams * wpfloat(2.0)  # (with ams*gam(bms+1.0_wp) where gam(3) = 2)
    LMD_0 = wpfloat(1.0e10)  # no snow value of lambda
    BX = wpfloat(1.0) / (GraupelConsts.bms + wpfloat(1.0))  # Exponent
    QSMIN = wpfloat(0.0e-6)  # TODO(): Check with Georgiana that this value is correct

    return where(qs > GraupelConsts.qmin, power((A2 * ns / ((qs + QSMIN) * rho)), BX), LMD_0)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def snow_lambda(
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    qs: fa.CellKField[ta.wpfloat],  # Snow specific mass
    ns: fa.CellKField[ta.wpfloat],  # Snow number
    riming_snow_rate: fa.CellKField[ta.wpfloat],  # output
):
    _snow_lambda(rho=rho, qs=qs, ns=ns, out=riming_snow_rate)


@gtx.field_operator
def _snow_number(
    t: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the snow number

    Args:
        t:            Temperature
        rho:          Ambient air density
        qs:           Snow specific mass

    Result:           Snow number
    """
    TMIN = ThermodynamicConsts.tmelt - wpfloat(40.0)
    TMAX = ThermodynamicConsts.tmelt
    QSMIN = wpfloat(2.0e-6)
    XA1 = wpfloat(-1.65e0)
    XA2 = wpfloat(5.45e-2)
    XA3 = wpfloat(3.27e-4)
    XB1 = wpfloat(1.42e0)
    XB2 = wpfloat(1.19e-2)
    XB3 = wpfloat(9.60e-5)
    N0S0 = wpfloat(8.00e5)
    N0S1 = wpfloat(13.5) * wpfloat(5.65e05)
    N0S2 = wpfloat(-0.107)
    N0S3 = wpfloat(13.5)
    N0S4 = wpfloat(0.5) * N0S1
    N0S5 = wpfloat(1.0e6)
    N0S6 = wpfloat(1.0e2) * N0S1
    N0S7 = wpfloat(1.0e9)

    # TODO(): see if these can be incorporated into WHERE statement
    tc = maximum(minimum(t, TMAX), TMIN) - ThermodynamicConsts.tmelt
    alf = power(wpfloat(10.0), (XA1 + tc * (XA2 + tc * XA3)))
    bet = XB1 + tc * (XB2 + tc * XB3)
    n0s = (
        N0S3
        * power(((qs + QSMIN) * rho / GraupelConsts.ams), (wpfloat(4.0) - wpfloat(3.0) * bet))
        / (alf * alf * alf)
    )
    y = exp(N0S2 * tc)
    n0smn = maximum(N0S4 * y, N0S5)
    n0smx = minimum(N0S6 * y, N0S7)
    return where(qs > GraupelConsts.qmin, minimum(n0smx, maximum(n0smn, n0s)), N0S0)


@gtx.field_operator
def _snow_number_scalar(
    t: ta.wpfloat,
    rho: ta.wpfloat,
    qs: ta.wpfloat,
) -> ta.wpfloat:
    """
    Compute the snow number

    Args:
        t:            Temperature
        rho:          Ambient air density
        qs:           Snow specific mass

    Result:           Snow number
    """
    TMIN = ThermodynamicConsts.tmelt - wpfloat(40.0)
    TMAX = ThermodynamicConsts.tmelt
    QSMIN = wpfloat(2.0e-6)
    XA1 = wpfloat(-1.65e0)
    XA2 = wpfloat(5.45e-2)
    XA3 = wpfloat(3.27e-4)
    XB1 = wpfloat(1.42e0)
    XB2 = wpfloat(1.19e-2)
    XB3 = wpfloat(9.60e-5)
    N0S0 = wpfloat(8.00e5)
    N0S1 = wpfloat(13.5) * wpfloat(5.65e05)
    N0S2 = wpfloat(-0.107)
    N0S3 = wpfloat(13.5)
    N0S4 = wpfloat(0.5) * N0S1
    N0S5 = wpfloat(1.0e6)
    N0S6 = wpfloat(1.0e2) * N0S1
    N0S7 = wpfloat(1.0e9)

    # TODO(): see if these can be incorporated into WHERE statement
    tc = maximum(minimum(t, TMAX), TMIN) - ThermodynamicConsts.tmelt
    alf = power(wpfloat(10.0), (XA1 + tc * (XA2 + tc * XA3)))
    bet = XB1 + tc * (XB2 + tc * XB3)
    n0s = (
        N0S3
        * power(((qs + QSMIN) * rho / GraupelConsts.ams), (wpfloat(4.0) - wpfloat(3.0) * bet))
        / (alf * alf * alf)
    )
    y = exp(N0S2 * tc)
    n0smn = maximum(N0S4 * y, N0S5)
    n0smx = minimum(N0S6 * y, N0S7)
    return minimum(n0smx, maximum(n0smn, n0s)) if qs > GraupelConsts.qmin else N0S0


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def snow_number(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    rho: fa.CellKField[ta.wpfloat],  # Ambient air density
    qs: fa.CellKField[ta.wpfloat],  # Snow specific mass
    number: fa.CellKField[ta.wpfloat],  # output
):
    _snow_number(t=t, rho=rho, qs=qs, out=number)


@gtx.field_operator
def _vel_scale_factor_ice(
    xrho: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the velocity scaling factor of ice

    Args:
        xrho:              sqrt(rho_00/rho)

    Result:                velocity scaling factor of ice
    """
    B_I = wpfloat(0.66666666666666667)
    return power(xrho, B_I)


@gtx.field_operator
def _vel_scale_factor_ice_scalar(
    xrho: ta.wpfloat,
) -> ta.wpfloat:
    """
    Compute the velocity scaling factor of ice

    Args:
        xrho:              sqrt(rho_00/rho)

    Result:                velocity scaling factor of ice
    """
    B_I = wpfloat(0.66666666666666667)
    return power(xrho, B_I)


@gtx.field_operator
def _vel_scale_factor_snow(
    xrho: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    t: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the velocity scaling factor of snow

    Args:
        xrho:              sqrt(rho_00/rho)
        rho:               Density of condensate
        t:                 Temperature
        qs:                Specific mass

    Result:                Velocity scaling factor of snow
    """
    B_S = wpfloat(-0.16666666666666667)
    return xrho * power(_snow_number(t, rho, qs), B_S)


@gtx.field_operator
def _vel_scale_factor_snow_scalar(
    xrho: ta.wpfloat,
    rho: ta.wpfloat,
    t: ta.wpfloat,
    qs: ta.wpfloat,
) -> ta.wpfloat:
    """
    Compute the velocity scaling factor of snow

    Args:
        xrho:              sqrt(rho_00/rho)
        rho:               Density of condensate
        t:                 Temperature
        qs:                Specific mass

    Result:                Velocity scaling factor of snow
    """
    B_S = wpfloat(-0.16666666666666667)
    return xrho * power(_snow_number_scalar(t=t, rho=rho, qs=qs), B_S)


@gtx.field_operator
def _vel_scale_factor_default(
    xrho: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the default velocity scaling factor

    Args:
        xrho:              sqrt(rho_00/rho)

    Result:                default velocity scaling factor
    """
    return xrho


@gtx.field_operator
def _vel_scale_factor_default_scalar(
    xrho: ta.wpfloat,
) -> ta.wpfloat:
    """
    Compute the default velocity scaling factor

    Args:
        xrho:              sqrt(rho_00/rho)

    Result:                default velocity scaling factor
    """
    return xrho


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def vel_scale_factor_ice(
    xrho: fa.CellKField[ta.wpfloat],  # sqrt(rho_00/rho)
    scale_factor: fa.CellKField[ta.wpfloat],  # output
):
    _vel_scale_factor_ice(xrho=xrho, out=scale_factor)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def vel_scale_factor_snow(
    xrho: fa.CellKField[ta.wpfloat],  # sqrt(rho_00/rho)
    rho: fa.CellKField[ta.wpfloat],  # Density of condensate
    t: fa.CellKField[ta.wpfloat],  # Temperature
    qs: fa.CellKField[ta.wpfloat],  # Specific mass
    scale_factor: fa.CellKField[ta.wpfloat],  # output
) -> None:
    _vel_scale_factor_snow(xrho=xrho, rho=rho, t=t, qs=qs, out=scale_factor)

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import exp, maximum, minimum, power, sqrt, where

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import (
    GraupelConsts,
    ThermodynamicConsts,
)
from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _cloud_to_graupel(
    t: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from cloud to graupel

    Args:
        t:                  Temperature
        rho:                Ambient density
        qc:                 Cloud specific mass
        qg:                 Graupel specific mass

    Return:                 Conversion rate
    """
    A_RIM = wpfloat(4.43)
    B_RIM = wpfloat(0.94878)
    ZERO = wpfloat(0.0)
    return where(
        (minimum(qc, qg) > GraupelConsts.qmin) & (t > GraupelConsts.tfrz_hom),
        A_RIM * qc * power(qg * rho, B_RIM),
        ZERO,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def cloud_to_graupel(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    qc: fa.CellKField[ta.wpfloat],  # Cloud specific mass
    qg: fa.CellKField[ta.wpfloat],  # Graupel specific mass
    riming_graupel_rate: fa.CellKField[ta.wpfloat],  # output
):
    _cloud_to_graupel(t=t, rho=rho, qc=qc, qg=qg, out=riming_graupel_rate)


@gtx.field_operator
def _cloud_to_rain(
    t: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    nc: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from cloud to rain

    Args:
        t:                  Temperature
        qc:                 Cloud specific mass
        qr:                 Rain water specific mass
        nc:                 Cloud water number concentration

    Return:                 Conversion rate
    """
    QMIN_AC = wpfloat(1.0e-6)  # threshold for auto conversion
    TAU_MAX = wpfloat(0.90e0)  # maximum allowed value of tau
    TAU_MIN = wpfloat(1.0e-30)  # minimum allowed value of tau
    A_PHI = wpfloat(6.0e2)  # constant in phi-function for autoconversion
    B_PHI = wpfloat(0.68e0)  # exponent in phi-function for autoconversion
    C_PHI = wpfloat(5.0e-5)  # exponent in phi-function for accretion
    AC_KERNEL = wpfloat(5.25e0)  # kernel coeff for SB2001 accretion
    X3 = wpfloat(2.0e0)  # gamma exponent for cloud distribution
    X2 = wpfloat(2.6e-10)  # separating mass between cloud and rain
    X1 = wpfloat(9.44e9)  # kernel coeff for SB2001 autoconversion
    ZERO = wpfloat(0.0)
    ONE = wpfloat(1.0)
    TWO = wpfloat(2.0)
    THREE = wpfloat(3.0)
    FOUR = wpfloat(4.0)
    AU_KERNEL = X1 / (wpfloat(20.0) * X2) * (X3 + TWO) * (X3 + FOUR) / ((X3 + ONE) * (X3 + ONE))

    # TO-DO: put as much of this into the WHERE statement as possible
    tau = maximum(TAU_MIN, minimum(ONE - qc / (qc + qr), TAU_MAX))  # temporary cannot go in where
    phi = power(tau, B_PHI)
    phi = A_PHI * phi * power(ONE - phi, THREE)
    xau = AU_KERNEL * power(qc * qc / nc, TWO) * (ONE + phi / power(ONE - tau, TWO))
    xac = AC_KERNEL * qc * qr * power(tau / (tau + C_PHI), FOUR)
    return where((qc > QMIN_AC) & (t > GraupelConsts.tfrz_hom), xau + xac, ZERO)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def cloud_to_rain(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    qc: fa.CellKField[ta.wpfloat],  # Cloud specific mass
    qr: fa.CellKField[ta.wpfloat],  # Rain water specific mass
    nc: ta.wpfloat,  # Cloud water number concentration
    conversion_rate: fa.CellKField[ta.wpfloat],  # output
):
    _cloud_to_rain(t=t, qc=qc, qr=qr, nc=nc, out=conversion_rate)


@gtx.field_operator
def _cloud_to_snow(
    t: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    ns: fa.CellKField[ta.wpfloat],
    lam: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from cloud to snow

    Args:
        t:                  Temperature
        qc:                 Cloud specific mass
        qs:                 Snow specific mass
        ns:                 Snow number
        lam:                Snow slope parameter (lambda)

    Return:                 Conversion rate
    """
    ECS = wpfloat(0.9)
    B_RIM = -(wpfloat(GraupelConsts.v1s) + wpfloat(3.0))
    C_RIM = wpfloat(2.61) * ECS * GraupelConsts.v0s  # (with pi*gam(v1s+3)/4 = 2.610)
    ZERO = wpfloat(0.0)
    return where(
        (minimum(qc, qs) > GraupelConsts.qmin) & (t > GraupelConsts.tfrz_hom),
        C_RIM * ns * qc * power(lam, B_RIM),
        ZERO,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def cloud_to_snow(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],  # Temperature
    qc: fa.CellKField[ta.wpfloat],  # Cloud specific mass
    qs: fa.CellKField[ta.wpfloat],  # Snow specific mass
    ns: fa.CellKField[ta.wpfloat],  # Snow number
    lam: fa.CellKField[ta.wpfloat],  # Snow slope parameter
    riming_snow_rate: fa.CellKField[ta.wpfloat],  # output
):
    _cloud_to_snow(t=t, qc=qc, qs=qs, ns=ns, lam=lam, out=riming_snow_rate)


@gtx.field_operator
def _cloud_x_ice(
    t: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from cloud to ice

    Args:
        t:                  Temperature
        qc:                 Cloud specific mass
        qi:                 Ice specific mass
        dt:                 Time step

    Return:                 Freezing rate
    """
    ZERO = wpfloat(0.0)
    result = where((qc > GraupelConsts.qmin) & (t < GraupelConsts.tfrz_hom), qc / dt, ZERO)
    result = where((qi > GraupelConsts.qmin) & (t > ThermodynamicConsts.tmelt), -qi / dt, result)
    return result


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def cloud_x_ice(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    qc: fa.CellKField[ta.wpfloat],  # Cloud specific mass
    qi: fa.CellKField[ta.wpfloat],  # Ice specific mass
    dt: ta.wpfloat,  # time step
    freezing_rate: fa.CellKField[ta.wpfloat],  # output
):
    _cloud_x_ice(t=t, qc=qc, qi=qi, dt=dt, out=freezing_rate)


@gtx.field_operator
def _graupel_to_rain(
    t: fa.CellKField[ta.wpfloat],
    p: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    dvsw0: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from graupel to rain

    Args:
        t:                  Temperature
        p:                  Ambient pressure
        rho:                Ambient density
        dvsw0:              qv-qsat_water(T0)
        qg:                 Graupel specific mass

    Return:                 Conversion rate
    """
    A_MELT = wpfloat(GraupelConsts.tx) - wpfloat(389.5)  # melting prefactor
    B_MELT = wpfloat(0.6)  # melting exponent
    C1_MELT = wpfloat(12.31698)  # Constants in melting formula
    C2_MELT = wpfloat(7.39441e-05)  # Constants in melting formula
    ZERO = wpfloat(0.0)
    return where(
        (
            t
            > maximum(
                ThermodynamicConsts.tmelt,
                ThermodynamicConsts.tmelt - GraupelConsts.tx * dvsw0,
            )
        )
        & (qg > GraupelConsts.qmin),
        (C1_MELT / p + C2_MELT)
        * (t - ThermodynamicConsts.tmelt + A_MELT * dvsw0)
        * power(qg * rho, B_MELT),
        ZERO,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def graupel_to_rain(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],  # Ambient temperature
    p: fa.CellKField[ta.wpfloat],  # Ambient pressue
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    dvsw0: fa.CellKField[ta.wpfloat],  # qv-qsat_water(T0)
    qg: fa.CellKField[ta.wpfloat],  # Graupel specific mass
    rain_rate: fa.CellKField[ta.wpfloat],  # output
):
    _graupel_to_rain(t=t, p=p, rho=rho, dvsw0=dvsw0, qg=qg, out=rain_rate)


@gtx.field_operator
def _ice_to_graupel(
    rho: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    sticking_eff: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from ice to graupel

    Args:
        rho:                Ambient density
        qr:                 Rain specific mass
        qg:                 Graupel specific mass
        qi:                 Ice specific mass
        sticking_eff:       Sticking efficiency

    Return:                 Conversion rate
    """
    A_CT = wpfloat(1.72)  # (15/32)*(PI**0.5)*(EIR/RHOW)*V0R*AR**(1/8)
    B_CT = wpfloat(0.875)  # Exponent = 7/8
    C_AGG_CT = wpfloat(2.46)
    B_AGG_CT = wpfloat(0.94878)  # Exponent
    ZERO = wpfloat(0.0)
    result = where(
        (qi > GraupelConsts.qmin) & (qg > GraupelConsts.qmin),
        sticking_eff * qi * C_AGG_CT * power(rho * qg, B_AGG_CT),
        ZERO,
    )
    result = where(
        (qi > GraupelConsts.qmin) & (qr > GraupelConsts.qmin),
        result + A_CT * qi * power(rho * qr, B_CT),
        result,
    )
    return result


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_to_graupel(  # noqa: PLR0917 [too-many-positional-arguments]
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    qr: fa.CellKField[ta.wpfloat],  # Rain specific mass
    qg: fa.CellKField[ta.wpfloat],  # Graupel specific mass
    qi: fa.CellKField[ta.wpfloat],  # Ice specific mass
    sticking_eff: fa.CellKField[ta.wpfloat],  # Sticking efficiency
    aggregation: fa.CellKField[ta.wpfloat],  # output
):
    _ice_to_graupel(rho=rho, qr=qr, qg=qg, qi=qi, sticking_eff=sticking_eff, out=aggregation)


@gtx.field_operator
def _ice_to_snow(
    qi: fa.CellKField[ta.wpfloat],
    ns: fa.CellKField[ta.wpfloat],
    lam: fa.CellKField[ta.wpfloat],
    sticking_eff: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from ice to snow

    Args:
        qi:                 Ice specific mass
        ns:                 Snow number
        lam:                Snow intercept parameter
        sticking_eff:       Sticking efficiency

    Return:                 Conversion rate
    """
    QI0 = wpfloat(0.0)  # Critical ice required for autoconversion
    C_IAU = wpfloat(1.0e-3)  # Coefficient of auto conversion
    C_AGG = wpfloat(2.61) * GraupelConsts.v0s  # Coeff of aggregation (2.610 = pi*gam(v1s+3)/4)
    B_AGG = -(GraupelConsts.v1s + wpfloat(3.0))  # Aggregation exponent
    ZERO = wpfloat(0.0)

    return where(
        (qi > GraupelConsts.qmin),
        sticking_eff * (C_IAU * maximum(ZERO, (qi - QI0)) + qi * (C_AGG * ns) * power(lam, B_AGG)),
        ZERO,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_to_snow(
    qi: fa.CellKField[ta.wpfloat],  # Ice specific mass
    ns: fa.CellKField[ta.wpfloat],  # Snow number
    lam: fa.CellKField[ta.wpfloat],  # Snow intercept parameter
    sticking_eff: fa.CellKField[ta.wpfloat],  # Sticking efficiency
    conversion_rate: fa.CellKField[ta.wpfloat],  # output
):
    _ice_to_snow(qi=qi, ns=ns, lam=lam, sticking_eff=sticking_eff, out=conversion_rate)


@gtx.field_operator
def _rain_to_graupel(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    mi: fa.CellKField[ta.wpfloat],
    dvsw: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from rain to graupel

    Args:
        t:                  Temperature
        rho:                Ambient density
        qc:                 Cloud specific mass
        qr:                 Rain specific mass
        qi:                 Ice specific mass
        qs:                 Snow specific mass
        mi:                 Ice crystal mass
        dvsw:               qv-qsat_water (T)
        dt:                 Time step

    Return:                 Conversion rate
    """
    TFRZ_RAIN = ThermodynamicConsts.tmelt - wpfloat(2.0)
    A1 = wpfloat(9.95e-5)  # coefficient for immersion raindrop freezing: alpha_if
    B1 = wpfloat(1.75)  # coefficient for immersion raindrop freezing: a_if
    #  C1 assigned to 1.68 in original but not used there; coefficient for raindrop freezing
    C2 = wpfloat(0.66)  # coefficient for immersion raindrop freezing: a_if
    C3 = wpfloat(1.0)  # coefficient for immersion raindrop freezing: a_if
    C4 = wpfloat(0.1)  # coefficient for immersion raindrop freezing: a_if
    A2 = wpfloat(1.24e-3)  # (PI/24)*EIR*V0R*Gamma(6.5)*AR**(-5/8)
    B2 = wpfloat(1.625)  # exponent for rho*qr
    QS_CRIT = wpfloat(1.0e-7)  # critical humidity of snow
    ZERO = wpfloat(0.0)

    maskinner = (dvsw + qc <= ZERO) | (qr > C4 * qc)
    mask = (qr > GraupelConsts.qmin) & (t < TFRZ_RAIN)
    result = where(
        mask & (t > GraupelConsts.tfrz_hom) & maskinner,
        (exp(C2 * (TFRZ_RAIN - t)) - C3) * (A1 * power((qr * rho), B1)),
        ZERO,
    )
    result = where(mask & (t <= GraupelConsts.tfrz_hom), qr / dt, result)

    return where(
        (minimum(qi, qr) > GraupelConsts.qmin) & (qs > QS_CRIT),
        result + A2 * (qi / mi) * power((rho * qr), B2),
        result,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def rain_to_graupel(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],  # Temperature
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    qc: fa.CellKField[ta.wpfloat],  # Cloud specific mass
    qr: fa.CellKField[ta.wpfloat],  # Specific humidity of rain
    qi: fa.CellKField[ta.wpfloat],  # Ice specific mass
    qs: fa.CellKField[ta.wpfloat],  # Snow specific mass
    mi: fa.CellKField[ta.wpfloat],  # Ice crystal mass
    dvsw: fa.CellKField[ta.wpfloat],  # qv-qsat_water (T)
    dt: ta.wpfloat,  # time step
    conversion_rate: fa.CellKField[ta.wpfloat],  # output
):
    _rain_to_graupel(
        t=t, rho=rho, qc=qc, qr=qr, qi=qi, qs=qs, mi=mi, dvsw=dvsw, dt=dt, out=conversion_rate
    )


@gtx.field_operator
def _rain_to_vapor(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    dvsw: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from rain to vapor

    Args:
        t:                  Temperature
        rho:                Ambient density
        qc:                 Cloud specific mass
        qr:                 Rain specific mass
        dvsw:               qv-qsat_water (T)
        dt:                 Time step

    Return:                 Conversion rate
    """
    B1_RV = wpfloat(0.16667)  # exponent in power-law relation for mass density
    B2_RV = wpfloat(0.55555)  # exponent for qr*rho
    C1_RV = wpfloat(0.61)  # coefficient for tc^0 in quadratic expansion
    C2_RV = wpfloat(-0.0163)  # coefficient of tc^1 in quadratic expansion
    C3_RV = wpfloat(1.111e-4)  # coefficient of tc^2 in quadratic expansion
    A1_RV = wpfloat(1.536e-3)  # coefficient 1 in qr reconstruction
    A2_RV = wpfloat(1.0e0)  # coefficient 2 in qr reconstruction
    A3_RV = wpfloat(19.0621e0)  # coefficient 3 in qr reconstruction
    ZERO = wpfloat(0.0)

    # TO-DO: move as much as possible into WHERE statement
    tc = t - ThermodynamicConsts.tmelt
    evap_max = (C1_RV + tc * (C2_RV + C3_RV * tc)) * (-dvsw) / dt
    return where(
        (qr > GraupelConsts.qmin) & (dvsw + qc <= ZERO),
        minimum(
            A1_RV * (A2_RV + A3_RV * power(qr * rho, B1_RV)) * (-dvsw) * power(qr * rho, B2_RV),
            evap_max,
        ),
        ZERO,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def rain_to_vapor(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],  # Temperature
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    qc: fa.CellKField[ta.wpfloat],  # Cloud-specific humidity
    qr: fa.CellKField[ta.wpfloat],  # Rain-specific humidity
    dvsw: fa.CellKField[ta.wpfloat],  # qv-qsat_water (T)
    dt: ta.wpfloat,  # time step
    conversion_rate: fa.CellKField[ta.wpfloat],  # output
):
    _rain_to_vapor(t=t, rho=rho, qc=qc, qr=qr, dvsw=dvsw, dt=dt, out=conversion_rate)


@gtx.field_operator
def _snow_to_graupel(
    t: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from snow to graupel

    Args:
        t:                  Temperature
        rho:                Ambient density
        qc:                 Cloud specific mass
        qs:                 Snow specific mass

    Return:                 Conversion rate
    """
    A_RIM_CT = wpfloat(0.5)  # Constants in riming formula
    B_RIM_CT = wpfloat(0.75)
    ZERO = wpfloat(0.0)
    return where(
        (minimum(qc, qs) > GraupelConsts.qmin) & (t > GraupelConsts.tfrz_hom),
        A_RIM_CT * qc * power(qs * rho, B_RIM_CT),
        ZERO,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def snow_to_graupel(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    qc: fa.CellKField[ta.wpfloat],  # Cloud specific mass
    qs: fa.CellKField[ta.wpfloat],  # Snow specific mass
    conversion_rate: fa.CellKField[ta.wpfloat],  # output
):
    _snow_to_graupel(t=t, rho=rho, qc=qc, qs=qs, out=conversion_rate)


@gtx.field_operator
def _snow_to_rain(
    t: fa.CellKField[ta.wpfloat],
    p: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    dvsw0: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from snow to rain

    Args:
        t:                  Temperature
        p:                  Ambient pressure
        rho:                Ambient density
        dvsw0:              qv-qsat_water (T0)
        qs:                 Snow specific mass

    Return:                 Conversion rate
    """
    C1_SR = wpfloat(79.6863)  # Constants in melting formula
    C2_SR = wpfloat(0.612654e-3)  # Constants in melting formula
    A_SR = GraupelConsts.tx - wpfloat(389.5)  # Melting prefactor
    B_SR = wpfloat(0.8)  # Melting exponent
    ZERO = wpfloat(0.0)
    return where(
        (
            t
            > maximum(
                ThermodynamicConsts.tmelt,
                ThermodynamicConsts.tmelt - GraupelConsts.tx * dvsw0,
            )
        )
        & (qs > GraupelConsts.qmin),
        (C1_SR / p + C2_SR)
        * (t - ThermodynamicConsts.tmelt + A_SR * dvsw0)
        * power(qs * rho, B_SR),
        ZERO,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def snow_to_rain(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],  # Temperature
    p: fa.CellKField[ta.wpfloat],  # Ambient pressure
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    dvsw0: fa.CellKField[ta.wpfloat],  # qv-qsat_water(T0)
    qs: fa.CellKField[ta.wpfloat],  # Snow specific mass
    conversion_rate: fa.CellKField[ta.wpfloat],  # output
):
    _snow_to_rain(t=t, p=p, rho=rho, dvsw0=dvsw0, qs=qs, out=conversion_rate)


@gtx.field_operator
def _vapor_x_graupel(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],
    p: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    dvsw: fa.CellKField[ta.wpfloat],
    dvsi: fa.CellKField[ta.wpfloat],
    dvsw0: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from vapor to graupel

    Args:
        t:                  Temperature
        p:                  Ambient pressure
        rho:                Ambient density
        qg:                 Graupel specific mass
        dvsw:               qv-qsat_water (T)
        dvsi:               qv-qsat_ice (T)
        dvsw0:              qv-qsat_water (T0)
        dt:                 Time step

    Return:                 Conversion rate
    """
    A1_VG = wpfloat(0.398561)
    A2_VG = wpfloat(-0.00152398)
    A3 = wpfloat(2554.99)
    A4 = wpfloat(2.6531e-7)
    A5 = wpfloat(0.153907)
    A6 = wpfloat(-7.86703e-07)
    A7 = wpfloat(0.0418521)
    A8 = wpfloat(-4.7524e-8)
    B_VG = wpfloat(0.6)
    ZERO = wpfloat(0.0)
    result = where(
        (t < ThermodynamicConsts.tmelt),
        (A1_VG + A2_VG * t + A3 / p + A4 * p) * dvsi * power(qg * rho, B_VG),
        where(
            (t > (ThermodynamicConsts.tmelt - GraupelConsts.tx * dvsw0)),
            (A5 + A6 * p) * minimum(ZERO, dvsw0) * power(qg * rho, B_VG),
            (A7 + A8 * p) * dvsw * power(qg * rho, B_VG),
        ),
    )
    return where(qg > GraupelConsts.qmin, maximum(result, -qg / dt), ZERO)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def vapor_x_graupel(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],  # Temperature
    p: fa.CellKField[ta.wpfloat],  # Ambient pressure
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    qg: fa.CellKField[ta.wpfloat],  # Graupel specific mass
    dvsw: fa.CellKField[ta.wpfloat],  # qv-qsat_water(T)
    dvsi: fa.CellKField[ta.wpfloat],  # qv-qsat_ice(T)
    dvsw0: fa.CellKField[ta.wpfloat],  # qv-qsat_water(T0)
    dt: ta.wpfloat,  # time step
    exchange_rate: fa.CellKField[ta.wpfloat],  # output
):
    _vapor_x_graupel(
        t=t, p=p, rho=rho, qg=qg, dvsw=dvsw, dvsi=dvsi, dvsw0=dvsw0, dt=dt, out=exchange_rate
    )


@gtx.field_operator
def _vapor_x_ice(  # noqa: PLR0917 [too-many-positional-arguments]
    qi: fa.CellKField[ta.wpfloat],
    mi: fa.CellKField[ta.wpfloat],
    eta: fa.CellKField[ta.wpfloat],
    dvsi: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from vapor to ice

    Args:
        qi:                 Ice specific mass
        mi:                 Ice crystal mass
        eta:                Deposition factor
        dvsi:               Vapor excess qv-qsat_ice(T)
        rho:                Ambient density
        dt:                 Time step

    Return:                 Conversion rate
    """
    AMI = wpfloat(130.0)  # Form factor for mass-size relation of cold ice
    B_EXP = wpfloat(-0.67)  # exp. for conv. (-1 + 0.33) of ice mass to sfc area
    A_FACT = wpfloat(4.0) * AMI ** (
        wpfloat(-1.0) / wpfloat(3.0)
    )  #  Is (1.0/5.065797019100886) * 4.0
    ZERO = wpfloat(0.0)
    # TO-DO: see if this can be folded into the WHERE statement
    result = (A_FACT * eta) * rho * qi * power(mi, B_EXP) * dvsi
    result = where(
        result > ZERO,
        minimum(result, dvsi / dt),
        maximum(maximum(result, dvsi / dt), -qi / dt),
    )
    return where(qi > GraupelConsts.qmin, result, ZERO)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def vapor_x_ice(  # noqa: PLR0917 [too-many-positional-arguments]
    qi: fa.CellKField[ta.wpfloat],  # Specific humidity of ice
    mi: fa.CellKField[ta.wpfloat],  # Ice crystal mass
    eta: fa.CellKField[ta.wpfloat],  # Deposition factor
    dvsi: fa.CellKField[ta.wpfloat],  # Vapor excess qv-qsat_ice(T)
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    dt: ta.wpfloat,  # time step
    vapor_deposition_rate: fa.CellKField[ta.wpfloat],  # output
):
    _vapor_x_ice(qi=qi, mi=mi, eta=eta, dvsi=dvsi, rho=rho, dt=dt, out=vapor_deposition_rate)


@gtx.field_operator
def _vapor_x_snow(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],
    p: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    ns: fa.CellKField[ta.wpfloat],
    lam: fa.CellKField[ta.wpfloat],
    eta: fa.CellKField[ta.wpfloat],
    ice_dep: fa.CellKField[ta.wpfloat],
    dvsw: fa.CellKField[ta.wpfloat],
    dvsi: fa.CellKField[ta.wpfloat],
    dvsw0: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from vapor to snow

    Args:
        t:                  Temperature
        p:                  Ambient pressure
        rho:                Ambient density
        qs:                 Snow specific mass
        ns:                 Snow number
        lam:                Slope parameter (lambda) snow
        eta:                Deposition factor
        ice_dep:            Limiter for vapor dep on snow
        dvsw:               qv-qsat_water (T)
        dvsi:               qv-qsat_ice(T)
        dvsw0:              qv-qsat_water(T0)
        dt:                 Time step

    Return:                 Conversion rate
    """
    NU = wpfloat(1.75e-5)  # kinematic viscosity of air
    A0_VS = wpfloat(1.0)
    A1_VS = wpfloat(0.4182) * sqrt(GraupelConsts.v0s / NU)
    A2_VS = -(GraupelConsts.v1s + wpfloat(1.0)) / wpfloat(2.0)
    EPS = wpfloat(1.0e-15)
    QS_LIM = wpfloat(1.0e-7)
    CNX = wpfloat(4.0)
    B_VS = wpfloat(0.8)
    C1_VS = wpfloat(31282.3)
    C2_VS = wpfloat(0.241897)
    C3_VS = wpfloat(0.28003)
    C4_VS = wpfloat(-0.146293e-6)
    ZERO = wpfloat(0.0)

    # See if this can be incorporated into WHERE statement
    result = where(
        (t < ThermodynamicConsts.tmelt),
        (CNX * ns * eta / rho) * (A0_VS + A1_VS * power(lam, A2_VS)) * dvsi / (lam * lam + EPS),
        ZERO,
    )

    # GZ: This mask>0 limitation, which was missing in the original graupel scheme,
    # is crucial for numerical stability in the tropics!
    # a meaningful distinction between cloud ice and snow
    result = where(
        (t < ThermodynamicConsts.tmelt) & (result > ZERO),
        minimum(result, dvsi / dt - ice_dep),
        result,
    )
    result = where((t < ThermodynamicConsts.tmelt) & (qs <= QS_LIM), minimum(result, ZERO), result)
    # ELSE section
    result = where(
        (t >= ThermodynamicConsts.tmelt)
        & (t > (ThermodynamicConsts.tmelt - GraupelConsts.tx * dvsw0)),
        (C1_VS / p + C2_VS) * minimum(ZERO, dvsw0) * power(qs * rho, B_VS),
        result,
    )
    result = where(
        (t >= ThermodynamicConsts.tmelt)
        & (t <= (ThermodynamicConsts.tmelt - GraupelConsts.tx * dvsw0)),
        (C3_VS + C4_VS * p) * dvsw * power(qs * rho, B_VS),
        result,
    )
    return where((qs > GraupelConsts.qmin), maximum(result, -qs / dt), ZERO)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def vapor_x_snow(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],  # Temperature
    p: fa.CellKField[ta.wpfloat],  # Ambient pressure
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    qs: fa.CellKField[ta.wpfloat],  # Snow specific mass
    ns: fa.CellKField[ta.wpfloat],  # Snow number
    lam: fa.CellKField[ta.wpfloat],  # Slope parameter (lambda) snow
    eta: fa.CellKField[ta.wpfloat],  # Deposition factor
    ice_dep: fa.CellKField[ta.wpfloat],  # Limiter for vapor dep on snow
    dvsw: fa.CellKField[ta.wpfloat],  # qv-qsat_water(T)
    dvsi: fa.CellKField[ta.wpfloat],  # qv-qsat_ice(T)
    dvsw0: fa.CellKField[ta.wpfloat],  # qv-qsat_water(T0)
    dt: ta.wpfloat,  # time step
    vapor_deposition_rate: fa.CellKField[ta.wpfloat],  # output
) -> None:
    _vapor_x_snow(
        t=t,
        p=p,
        rho=rho,
        qs=qs,
        ns=ns,
        lam=lam,
        eta=eta,
        ice_dep=ice_dep,
        dvsw=dvsw,
        dvsi=dvsi,
        dvsw0=dvsw0,
        dt=dt,
        out=vapor_deposition_rate,
    )

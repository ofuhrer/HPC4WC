# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import exp

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import (
    GraupelConsts,
    ThermodynamicConsts,
)
from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _T_from_internal_energy(  # noqa: PLR0917 [too-many-positional-arguments]
    u: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qliq: fa.CellKField[ta.wpfloat],
    qice: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    dz: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the temperature from the internal energy

    Args:
        u:                  Internal energy (extensive)
        qv:                 Water vapor specific humidity
        qliq:               Specific mass of liquid phases
        qice:               Specific mass of solid phases
        rho:                Ambient density
        dz:                 Extent of grid cell

    Return:                 Temperature
    """
    qtot = qliq + qice + qv  # total water specific mass
    cv = (
        (
            ThermodynamicConsts.cvd * (wpfloat(1.0) - qtot)
            + ThermodynamicConsts.cvv * qv
            + ThermodynamicConsts.clw * qliq
            + GraupelConsts.ci * qice
        )
        * rho
        * dz
    )  # Moist isometric specific heat

    return (u + rho * dz * (qliq * GraupelConsts.lvc + qice * GraupelConsts.lsc)) / cv


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def T_from_internal_energy(  # noqa: PLR0917 [too-many-positional-arguments]
    u: fa.CellKField[ta.wpfloat],  # Internal energy (extensive)
    qv: fa.CellKField[ta.wpfloat],  # Water vapor specific humidity
    qliq: fa.CellKField[ta.wpfloat],  # Specific mass of liquid phases
    qice: fa.CellKField[ta.wpfloat],  # Specific mass of solid phases
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    dz: fa.CellKField[ta.wpfloat],  # Extent of grid cell
    temperature: fa.CellKField[ta.wpfloat],  # output
):
    _T_from_internal_energy(u=u, qv=qv, qliq=qliq, qice=qice, rho=rho, dz=dz, out=temperature)


@gtx.field_operator
def _T_from_internal_energy_scalar(  # noqa: PLR0917 [too-many-positional-arguments]
    u: ta.wpfloat,
    qv: ta.wpfloat,
    qliq: ta.wpfloat,
    qice: ta.wpfloat,
    rho: ta.wpfloat,
    dz: ta.wpfloat,
) -> ta.wpfloat:
    """
    Compute the temperature from the internal energy (scalar version callable from scan_operator)

    Args:
        u:                  Internal energy (extensive)
        qv:                 Water vapor specific humidity
        qliq:               Specific mass of liquid phases
        qice:               Specific mass of solid phases
        rho:                Ambient density
        dz:                 Extent of grid cell

    Return:                 Temperature
    """
    qtot = qliq + qice + qv  # total water specific mass
    cv = (
        (
            ThermodynamicConsts.cvd * (wpfloat(1.0) - qtot)
            + ThermodynamicConsts.cvv * qv
            + ThermodynamicConsts.clw * qliq
            + GraupelConsts.ci * qice
        )
        * rho
        * dz
    )  # Moist isometric specific heat

    return (u + rho * dz * (qliq * GraupelConsts.lvc + qice * GraupelConsts.lsc)) / cv


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def T_from_internal_energy_scalar(  # noqa: PLR0917 [too-many-positional-arguments]
    u: ta.wpfloat,  # Internal energy (extensive)
    qv: ta.wpfloat,  # Water vapor specific humidity
    qliq: ta.wpfloat,  # Specific mass of liquid phases
    qice: ta.wpfloat,  # Specific mass of solid phases
    rho: ta.wpfloat,  # Ambient density
    dz: ta.wpfloat,  # Extent of grid cell
    temperature: ta.wpfloat,  # output
):
    _T_from_internal_energy_scalar(
        u=u, qv=qv, qliq=qliq, qice=qice, rho=rho, dz=dz, out=temperature
    )


@gtx.field_operator
def _internal_energy(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qliq: fa.CellKField[ta.wpfloat],
    qice: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    dz: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the internal energy from the temperature

    Args:
        t:                 Temperature
        qv:                Specific mass of vapor
        qliq:              Specific mass of liquid phases
        qice:              Specific mass of solid phases
        rho:               Ambient density
        dz:                Extent of grid cell

    Result:                Internal energy
    """
    qtot = qliq + qice + qv
    cv = (
        ThermodynamicConsts.cvd * (wpfloat(1.0) - qtot)
        + ThermodynamicConsts.cvv * qv
        + ThermodynamicConsts.clw * qliq
        + GraupelConsts.ci * qice
    )

    return rho * dz * (cv * t - qliq * GraupelConsts.lvc - qice * GraupelConsts.lsc)


@gtx.field_operator
def _internal_energy_scalar(  # noqa: PLR0917 [too-many-positional-arguments]
    t: ta.wpfloat,
    qv: ta.wpfloat,
    qliq: ta.wpfloat,
    qice: ta.wpfloat,
    rho: ta.wpfloat,
    dz: ta.wpfloat,
) -> ta.wpfloat:
    """
    Compute the internal energy from the temperature

    Args:
        t:                 Temperature
        qv:                Specific mass of vapor
        qliq:              Specific mass of liquid phases
        qice:              Specific mass of solid phases
        rho:               Ambient density
        dz:                Extent of grid cell

    Result:                Internal energy
    """
    qtot = qliq + qice + qv
    cv = (
        ThermodynamicConsts.cvd * (wpfloat(1.0) - qtot)
        + ThermodynamicConsts.cvv * qv
        + ThermodynamicConsts.clw * qliq
        + GraupelConsts.ci * qice
    )

    return rho * dz * (cv * t - qliq * GraupelConsts.lvc - qice * GraupelConsts.lsc)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def internal_energy(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],  # Temperature
    qv: fa.CellKField[ta.wpfloat],  # Specific mass of vapor
    qliq: fa.CellKField[ta.wpfloat],  # Specific mass of liquid phases
    qice: fa.CellKField[ta.wpfloat],  # Specific mass of solid phases
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    dz: fa.CellKField[ta.wpfloat],  # Extent of grid cell
    energy: fa.CellKField[ta.wpfloat],  # output
):
    _internal_energy(t=t, qv=qv, qliq=qliq, qice=qice, rho=rho, dz=dz, out=energy)


@gtx.field_operator
def _qsat_ice_rho(
    t: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the Qsat pressure due to ice

    Args:
        t:                  Temperature
        rho:                Density

    Result:                 Pressure
    """
    C1ES = wpfloat(610.78)
    C3IES = wpfloat(21.875)
    C4IES = wpfloat(7.66)

    return (C1ES * exp(C3IES * (t - ThermodynamicConsts.tmelt) / (t - C4IES))) / (
        rho * ThermodynamicConsts.rv * t
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def qsat_ice_rho(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    rho: fa.CellKField[ta.wpfloat],  # Density
    pressure: fa.CellKField[ta.wpfloat],  # output
):
    _qsat_ice_rho(t=t, rho=rho, out=pressure)


@gtx.field_operator
def _qsat_rho(
    t: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the standard Qsat pressure

    Args:
        t:                  Temperature
        rho:                Density

    Result:                 Pressure
    """
    C1ES = wpfloat(610.78)
    C3LES = wpfloat(17.269)
    C4LES = wpfloat(35.86)

    return (C1ES * exp(C3LES * (t - ThermodynamicConsts.tmelt) / (t - C4LES))) / (
        rho * ThermodynamicConsts.rv * t
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def qsat_rho(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    rho: fa.CellKField[ta.wpfloat],  # Density
    pressure: fa.CellKField[ta.wpfloat],  # output
):
    _qsat_rho(t=t, rho=rho, out=pressure)


@gtx.field_operator
def _qsat_rho_tmelt(
    rho: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the Qsat pressure at t.melt

    Args:
        rho:                Density

    Result:                 Pressure
    """
    C1ES = wpfloat(610.78)

    return C1ES / (rho * ThermodynamicConsts.rv * ThermodynamicConsts.tmelt)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def qsat_rho_tmelt(
    rho: fa.CellKField[ta.wpfloat],  # Density
    pressure: fa.CellKField[ta.wpfloat],  # output
):
    _qsat_rho_tmelt(rho=rho, out=pressure)


@gtx.field_operator
def _dqsatdT_rho(
    qs: fa.CellKField[ta.wpfloat],
    t: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the derivative of saturation vapor pressure (over liquid)

    Args:
        qs:                 Saturation vapor pressure (over liquid)
        t:                  Temperature

    Result:                 Derivative d(qsat_rho)/dT
    """
    C3LES = wpfloat(17.269)
    C4LES = wpfloat(35.86)
    C5LES = C3LES * (ThermodynamicConsts.tmelt - C4LES)

    return qs * (C5LES / ((t - C4LES) * (t - C4LES)) - wpfloat(1.0) / t)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def dqsatdT_rho(
    qs: fa.CellKField[ta.wpfloat],  # Saturation vapor pressure (over liquid)
    t: fa.CellKField[ta.wpfloat],  # Temperature
    derivative: fa.CellKField[ta.wpfloat],  # output
):
    _dqsatdT_rho(qs=qs, t=t, out=derivative)


@gtx.field_operator
def _sat_pres_ice(
    t: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the saturation pressure over ice

    Args:
        t:                    Temperature

    Result:                   Saturation pressure
    """
    C1ES = wpfloat(610.78)
    C3IES = wpfloat(21.875)
    C4IES = wpfloat(7.66)

    return C1ES * exp(C3IES * (t - ThermodynamicConsts.tmelt) / (t - C4IES))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def sat_pres_ice(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    pressure: fa.CellKField[ta.wpfloat],  # output
):
    _sat_pres_ice(t=t, out=pressure)


@gtx.field_operator
def _sat_pres_water(
    t: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the saturation pressure over water

    Args:
        t:                    Temperature

    Result:                   Saturation pressure
    """
    C1ES = wpfloat(610.78)
    C3LES = wpfloat(17.269)
    C4LES = wpfloat(35.86)

    return C1ES * exp(C3LES * (t - ThermodynamicConsts.tmelt) / (t - C4LES))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def sat_pres_water(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    pressure: fa.CellKField[ta.wpfloat],  # output
) -> None:
    _sat_pres_water(t=t, out=pressure)


@gtx.field_operator
def _newton_raphson(  # noqa: PLR0917 [too-many-positional-arguments]
    Tx: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qve: fa.CellKField[ta.wpfloat],
    qce: fa.CellKField[ta.wpfloat],
    cvc: fa.CellKField[ta.wpfloat],
    ue: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute a Newton-Raphson iteration on the temperature

    Args:
        Tx:                    Temperature
        rho:                   Density containing dry air and water constituents
        qve:                   Specific humidity
        qre:                   Specific rain water
        cvc:                   Constant volume precalculation
        ue:                    Energy

    Result:                    Revised temperature
    """
    qx = _qsat_rho(Tx, rho)
    dqx = _dqsatdT_rho(qx, Tx)
    qcx = qve + qce - qx
    cv = cvc + ThermodynamicConsts.cvv * qx + ThermodynamicConsts.clw * qcx
    ux = cv * Tx - qcx * GraupelConsts.lvc
    dux = cv + dqx * (GraupelConsts.lvc + (ThermodynamicConsts.cvv - ThermodynamicConsts.clw) * Tx)
    Tx = Tx - (ux - ue) / dux
    return Tx

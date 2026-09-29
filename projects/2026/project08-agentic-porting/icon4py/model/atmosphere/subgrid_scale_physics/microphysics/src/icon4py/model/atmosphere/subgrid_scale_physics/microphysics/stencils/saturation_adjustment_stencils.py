# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import abs, maximum, where  # noqa: A004

import icon4py.model.common.dimension as dims
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics.stencils.microphysical_processes import (
    dqsatdT_rho,
    latent_heat_vaporization,
    qsat_rho,
)
from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.constants import PhysicsConstants


@gtx.field_operator
def _new_temperature_in_newton_iteration(
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    lwdocvd: fa.CellKField[ta.wpfloat],
    next_temperature: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Update the temperature in saturation adjustment by Newton iteration. Moist enthalpy and mass are conserved.
    The latent heat is assumed to be constant with its value computed from the initial temperature.
        T + Lv / cvd qv = TH, qv + qc = QTOT
        T = TH - Lv / cvd qsat(T), which is a transcendental function. Newton method is applied to solve it for T.
        f(T) = Lv / cvd qsat(T) + T - TH
        f'(T) = Lv / cvd dqsat(T)/dT + 1
        T_new = T - f(T)/f'(T) = ( TH - Lv / cvd (qsat(T) - T dqsat(T)/dT) ) / (Lv / cvd dqsat(T)/dT + 1)

    Args:
        temperature: initial temperature [K]
        qv: specific humidity [kg kg-1]
        rho: total air density [kg m-3]
        lwdocvd: Lv / cvd [K]
        next_temperature: temperature at previous iteration [K]
    Returns:
        updated temperature [K]
    """
    ft = next_temperature - temperature + lwdocvd * (qsat_rho(next_temperature, rho) - qv)
    dft = 1.0 + lwdocvd * dqsatdT_rho(next_temperature, qsat_rho(next_temperature, rho))

    return next_temperature - ft / dft


@gtx.field_operator
def _update_temperature_by_newton_iteration(
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    newton_iteration_mask: fa.CellKField[bool],
    lwdocvd: fa.CellKField[ta.wpfloat],
    next_temperature: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    current_temperature = where(
        newton_iteration_mask,
        _new_temperature_in_newton_iteration(
            temperature=temperature,
            qv=qv,
            rho=rho,
            lwdocvd=lwdocvd,
            next_temperature=next_temperature,
        ),
        next_temperature,
    )
    return current_temperature


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_temperature_by_newton_iteration(
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    newton_iteration_mask: fa.CellKField[bool],
    lwdocvd: fa.CellKField[ta.wpfloat],
    next_temperature: fa.CellKField[ta.wpfloat],
    current_temperature: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_temperature_by_newton_iteration(
        temperature=temperature,
        qv=qv,
        rho=rho,
        newton_iteration_mask=newton_iteration_mask,
        lwdocvd=lwdocvd,
        next_temperature=next_temperature,
        out=current_temperature,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _update_temperature_qv_qc_tendencies(
    dtime: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    current_temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    subsaturated_mask: fa.CellKField[bool],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    """
    Compute temperature, qv, and qc tendencies from the saturation adjustment.

    Args:
        dtime: time step
        temperature: initial temperature [K]
        current_temperature: temperature updated by saturation adjustment [K]
        qv: initial specific humidity [kg kg-1]
        qc: initial cloud mixing ratio [kg kg-1]
        rho: total air density [kg m-3]
        subsaturated_mask: a mask where the air is subsaturated even if all cloud particles evaporate
    Returns:
        (updated temperature - initial temperautre) / dtime [K s-1],
        (saturated specific humidity - initial specific humidity) / dtime [s-1],
        (total specific mixing ratio - saturated specific humidity - initial cloud specific mixing ratio) / dtime [s-1],
    """
    zqwmin = 1e-20
    qv_tendency, qc_tendency = where(
        subsaturated_mask,
        (qc / dtime, -qc / dtime),
        (
            (qsat_rho(current_temperature, rho) - qv) / dtime,
            (maximum(qv + qc - qsat_rho(current_temperature, rho), zqwmin) - qc) / dtime,
        ),
    )
    return (current_temperature - temperature) / dtime, qv_tendency, qc_tendency


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_temperature_qv_qc_tendencies(
    dtime: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    current_temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    subsaturated_mask: fa.CellKField[bool],
    temperature_tendency: fa.CellKField[ta.wpfloat],
    qv_tendency: fa.CellKField[ta.wpfloat],
    qc_tendency: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_temperature_qv_qc_tendencies(
        dtime=dtime,
        temperature=temperature,
        current_temperature=current_temperature,
        qv=qv,
        qc=qc,
        rho=rho,
        subsaturated_mask=subsaturated_mask,
        out=(temperature_tendency, qv_tendency, qc_tendency),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_subsaturated_case_and_initialize_newton_iterations(
    tolerance: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
) -> tuple[
    fa.CellKField[bool],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[bool],
]:
    """
    Preparation for saturation adjustment.
    First obtain the subsaturated case, where the saturation specific humidity is larger than qv + qc. This can be
    derived by computing saturation specific humidity at the temperature after all cloud particles are evaporated.
    If this is the case, the new temperature is simply the temperature after all cloud particles are evaporated, and
    qv_new = qv + qc, qc = 0.
    All the remaining grid cells are marked as newton_iteration_mask for which Newton iteration is required to solve
    for the new temperature.

    Args:
        tolerance: tolerance for convergence in Newton iteration
        temperature: initial temperature [K]
        qv: initial specific humidity [kg kg-1]
        qc: initial cloud mixing ratio [kg kg-1]
        rho: total air density [kg m-3]
    Returns:
        mask for subsaturated case,
        Lv / cvd,
        current temperature for starting the Newton iteration,
        next temperature for starting the Newton iteration,
        mask for Newton iteration case
    """
    temperature_after_all_qc_evaporated = (
        temperature - latent_heat_vaporization(temperature) / PhysicsConstants.cvd * qc
    )

    # Check, which points will still be subsaturated even after evaporating all cloud water.
    subsaturated_mask = qv + qc <= qsat_rho(temperature_after_all_qc_evaporated, rho)

    # Remains const. during iteration
    lwdocvd = latent_heat_vaporization(temperature) / PhysicsConstants.cvd

    current_temperature = where(
        subsaturated_mask,
        temperature_after_all_qc_evaporated,
        temperature - 2.0 * tolerance,
    )
    next_temperature = where(subsaturated_mask, temperature_after_all_qc_evaporated, temperature)
    newton_iteration_mask = where(subsaturated_mask, False, True)

    return subsaturated_mask, lwdocvd, current_temperature, next_temperature, newton_iteration_mask


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_subsaturated_case_and_initialize_newton_iterations(
    tolerance: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    subsaturated_mask: fa.CellKField[bool],
    lwdocvd: fa.CellKField[ta.wpfloat],
    current_temperature: fa.CellKField[ta.wpfloat],
    next_temperature: fa.CellKField[ta.wpfloat],
    newton_iteration_mask: fa.CellKField[bool],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_subsaturated_case_and_initialize_newton_iterations(
        tolerance=tolerance,
        temperature=temperature,
        qv=qv,
        qc=qc,
        rho=rho,
        out=(
            subsaturated_mask,
            lwdocvd,
            current_temperature,
            next_temperature,
            newton_iteration_mask,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_newton_iteration_mask_and_copy_temperature_on_converged_cells(
    tolerance: ta.wpfloat,
    current_temperature: fa.CellKField[ta.wpfloat],
    next_temperature: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[bool], fa.CellKField[ta.wpfloat]]:
    """
    Compute a mask for the next Newton iteration when the difference between new and old temperature is larger
    than the tolerance.
    Then, copy temperature from the current to the new temperature field where the convergence criterion is already met.
    Otherwise, it is zero (the value does not matter because it will be updated in next iteration).

    Args:
        tolerance: tolerance for convergence in Newton iteration
        current_temperature: temperature at previous Newtion iteration [K]
        next_temperature: temperature  at current Newtion iteration [K]
    Returns:
        new temperature [K]
    """
    newton_iteration_mask = where(
        abs(current_temperature - next_temperature) > tolerance, True, False
    )
    new_temperature = where(newton_iteration_mask, 0.0, current_temperature)
    return newton_iteration_mask, new_temperature


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_newton_iteration_mask_and_copy_temperature_on_converged_cells(
    tolerance: ta.wpfloat,
    current_temperature: fa.CellKField[ta.wpfloat],
    next_temperature: fa.CellKField[ta.wpfloat],
    newton_iteration_mask: fa.CellKField[bool],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_newton_iteration_mask_and_copy_temperature_on_converged_cells(
        tolerance=tolerance,
        current_temperature=current_temperature,
        next_temperature=next_temperature,
        out=(newton_iteration_mask, next_temperature),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

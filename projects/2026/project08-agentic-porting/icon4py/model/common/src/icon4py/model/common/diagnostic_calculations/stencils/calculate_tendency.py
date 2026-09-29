# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _calculate_virtual_temperature_tendency(
    dtime: ta.wpfloat,
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Update virtual temperature tendency.

    Args:
        dtime: time step [s]
        qv: specific humidity [kg kg-1]
        qc: specific cloud water content [kg kg-1]
        qi: specific cloud ice content [kg kg-1]
        qr: specific rain water content [kg kg-1]
        qs: specific snow content [kg kg-1]
        qg: specific graupel content [kg kg-1]
        temperature: air temperature [K]
        virtual_temperature: air virtual temperature [K]
    Returns:
        virtual temperature tendency [K s-1], exner tendency [s-1], new exner, new virtual temperature [K]
    """
    qsum = qc + qi + qr + qs + qg

    new_virtual_temperature = temperature * (
        wpfloat("1.0") + PhysicsConstants.rv_o_rd_minus_1 * qv - qsum
    )

    return (new_virtual_temperature - virtual_temperature) / dtime


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_virtual_temperature_tendency(
    dtime: ta.wpfloat,
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature_tendency: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _calculate_virtual_temperature_tendency(
        dtime=dtime,
        qv=qv,
        qc=qc,
        qi=qi,
        qr=qr,
        qs=qs,
        qg=qg,
        temperature=temperature,
        virtual_temperature=virtual_temperature,
        out=virtual_temperature_tendency,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _calculate_exner_tendency(
    dtime: ta.wpfloat,
    virtual_temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature_tendency: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Update exner tendency.

    Args:
        dtime: time step [s]
        virtual_temperature: air virtual temperature [K]
        virtual_temperature_tendency: air virtual temperature tendency [K s-1]
        exner: exner function
    Returns:
        exner tendency [s-1]
    """

    new_exner = exner * (
        wpfloat("1.0")
        + PhysicsConstants.rd_o_cpd * virtual_temperature_tendency / virtual_temperature * dtime
    )

    return (new_exner - exner) / dtime


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_exner_tendency(
    dtime: ta.wpfloat,
    virtual_temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature_tendency: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
    exner_tendency: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _calculate_exner_tendency(
        dtime=dtime,
        virtual_temperature=virtual_temperature,
        virtual_temperature_tendency=virtual_temperature_tendency,
        exner=exner,
        out=exner_tendency,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _calculate_cell_kdim_field_tendency(
    dtime: ta.wpfloat,
    old_field: fa.CellKField[ta.wpfloat],
    new_field: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Update tendency of Cell-K Dim field.

    Args:
        dtime: time step [s]
        old_field: any old Cell-K Dim field [unit]
        new_field: any new Cell-K Dim field [unit]
    Returns:
        tendency [unit s-1]
    """

    return (new_field - old_field) / dtime


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_cell_kdim_field_tendency(
    dtime: ta.wpfloat,
    old_field: fa.CellKField[ta.wpfloat],
    new_field: fa.CellKField[ta.wpfloat],
    tendency: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _calculate_cell_kdim_field_tendency(
        dtime=dtime,
        old_field=old_field,
        new_field=new_field,
        out=tendency,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

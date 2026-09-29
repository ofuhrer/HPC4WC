# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import vpfloat


@gtx.field_operator
def _compute_first_vertical_derivative_at_cells(
    cell_kdim_field: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """
    This stencil computes the first vertical at cells
    """
    first_vertical_derivative = (cell_kdim_field - cell_kdim_field(KDim + 1)) * inv_ddqz_z_full
    return first_vertical_derivative


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_first_vertical_derivative_at_cells(  # noqa: PLR0917 [too-many-positional-arguments]
    cell_kdim_field: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    first_vertical_derivative: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_first_vertical_derivative_at_cells(
        cell_kdim_field=cell_kdim_field,
        inv_ddqz_z_full=inv_ddqz_z_full,
        out=first_vertical_derivative,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

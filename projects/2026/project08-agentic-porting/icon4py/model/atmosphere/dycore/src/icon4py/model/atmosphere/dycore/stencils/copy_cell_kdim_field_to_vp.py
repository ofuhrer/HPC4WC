# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _copy_cell_kdim_field_to_vp(
    field: fa.CellKField[wpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_11 or _mo_solve_nonhydro_stencil_59."""
    field_copy = astype(field, vpfloat)
    return field_copy


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def copy_cell_kdim_field_to_vp(
    field: fa.CellKField[wpfloat],
    field_copy: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _copy_cell_kdim_field_to_vp(
        field=field,
        out=field_copy,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

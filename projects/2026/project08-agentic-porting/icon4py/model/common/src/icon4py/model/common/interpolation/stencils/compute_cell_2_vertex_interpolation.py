# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import neighbor_sum

import icon4py.model.common.type_alias as types
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import V2C, V2CDim


@gtx.field_operator
def _compute_cell_2_vertex_interpolation(
    cell_in: fa.CellKField[types.wpfloat],
    c_int: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], types.wpfloat],
) -> fa.VertexKField[types.wpfloat]:
    vert_out = neighbor_sum(c_int * cell_in(V2C), axis=V2CDim)
    return vert_out


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_cell_2_vertex_interpolation(
    cell_in: fa.CellKField[types.wpfloat],
    c_int: gtx.Field[[dims.VertexDim, dims.V2CDim], types.wpfloat],
    vert_out: fa.VertexKField[types.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    Compute the interpolation from cell to vertex field.

    Args:
        cell_in: input cell field
        c_int: interpolation coefficients
        vert_out: (output) vertex field
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_cell_2_vertex_interpolation(
        cell_in=cell_in,
        c_int=c_int,
        out=vert_out,
        domain={
            dims.VertexDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import V2C, V2CDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    p_cell_in: fa.CellKField[wpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, V2CDim], wpfloat],
) -> fa.VertexKField[vpfloat]:
    p_vert_out_wp = neighbor_sum(p_cell_in(V2C) * c_intp, axis=V2CDim)
    return astype(p_vert_out_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    p_cell_in: fa.CellKField[wpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, V2CDim], wpfloat],
    p_vert_out: fa.VertexKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
        p_cell_in=p_cell_in,
        c_intp=c_intp,
        out=p_vert_out,
        domain={
            dims.VertexDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

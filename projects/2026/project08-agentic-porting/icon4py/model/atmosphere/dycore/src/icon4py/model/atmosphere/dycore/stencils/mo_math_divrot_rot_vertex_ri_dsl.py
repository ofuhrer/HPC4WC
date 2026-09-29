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
from icon4py.model.common.dimension import V2E, V2EDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _mo_math_divrot_rot_vertex_ri_dsl(
    vec_e: fa.EdgeKField[wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, V2EDim], wpfloat],
) -> fa.VertexKField[vpfloat]:
    rot_vec_wp = neighbor_sum(vec_e(V2E) * geofac_rot, axis=V2EDim)
    return astype(rot_vec_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def mo_math_divrot_rot_vertex_ri_dsl(
    vec_e: fa.EdgeKField[wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, V2EDim], wpfloat],
    rot_vec: fa.VertexKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _mo_math_divrot_rot_vertex_ri_dsl(
        vec_e=vec_e,
        geofac_rot=geofac_rot,
        out=rot_vec,
        domain={
            dims.VertexDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

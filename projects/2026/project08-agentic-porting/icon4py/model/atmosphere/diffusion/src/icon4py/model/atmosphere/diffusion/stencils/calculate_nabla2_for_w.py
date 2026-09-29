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
from icon4py.model.common.dimension import C2E2CO, C2E2CODim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _calculate_nabla2_for_w(
    w: fa.CellKField[wpfloat], geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat]
) -> fa.CellKField[vpfloat]:
    z_nabla2_c_wp = neighbor_sum(w(C2E2CO) * geofac_n2s, axis=C2E2CODim)
    return astype(z_nabla2_c_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_nabla2_for_w(
    w: fa.CellKField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    z_nabla2_c: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    # TODO(): replace this by common/math/stencils/compute_nabla2_on_cell_k
    _calculate_nabla2_for_w(
        w=w,
        geofac_n2s=geofac_n2s,
        out=z_nabla2_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

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
def cell_horizontal_gradients_by_green_gauss_method(
    scalar_field: fa.CellKField[vpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    scalar_field_wp = astype((scalar_field), wpfloat)

    p_grad_1_u_wp = neighbor_sum(geofac_grg_x * scalar_field_wp(C2E2CO), axis=C2E2CODim)
    p_grad_1_v_wp = neighbor_sum(geofac_grg_y * scalar_field_wp(C2E2CO), axis=C2E2CODim)
    return astype((p_grad_1_u_wp, p_grad_1_v_wp), vpfloat)

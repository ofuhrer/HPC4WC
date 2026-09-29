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
from icon4py.model.common.dimension import C2E, C2EDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _add_interpolated_horizontal_advection_of_w(
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    z_v_grad_w: fa.EdgeKField[vpfloat],
    ddt_w_adv: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_17."""
    z_v_grad_w_wp, ddt_w_adv_wp = astype((z_v_grad_w, ddt_w_adv), wpfloat)
    ddt_w_adv_wp = ddt_w_adv_wp + neighbor_sum(z_v_grad_w_wp(C2E) * e_bln_c_s, axis=C2EDim)
    return astype(ddt_w_adv_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def add_interpolated_horizontal_advection_of_w(
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    z_v_grad_w: fa.EdgeKField[vpfloat],
    ddt_w_adv: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _add_interpolated_horizontal_advection_of_w(
        e_bln_c_s=e_bln_c_s,
        z_v_grad_w=z_v_grad_w,
        ddt_w_adv=ddt_w_adv,
        out=ddt_w_adv,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

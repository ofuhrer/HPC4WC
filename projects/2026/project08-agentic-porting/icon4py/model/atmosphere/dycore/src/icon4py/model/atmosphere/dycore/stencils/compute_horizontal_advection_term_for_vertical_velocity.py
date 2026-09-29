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
from icon4py.model.common.dimension import E2C, E2V
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_horizontal_advection_term_for_vertical_velocity(
    vn_ie: fa.EdgeKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    w: fa.CellKField[wpfloat],
    z_vt_ie: fa.EdgeKField[vpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    z_w_v: fa.VertexKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly know as _mo_velocity_advection_stencil_07."""
    z_vt_ie_wp, vn_ie_wp = astype((z_vt_ie, vn_ie), wpfloat)

    z_v_grad_w_wp = vn_ie_wp * inv_dual_edge_length * (
        w(E2C[0]) - w(E2C[1])
    ) + z_vt_ie_wp * inv_primal_edge_length * tangent_orientation * astype(
        z_w_v(E2V[0]) - z_w_v(E2V[1]), wpfloat
    )
    return astype(z_v_grad_w_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_horizontal_advection_term_for_vertical_velocity(
    vn_ie: fa.EdgeKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    w: fa.CellKField[wpfloat],
    z_vt_ie: fa.EdgeKField[vpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    z_w_v: fa.VertexKField[vpfloat],
    z_v_grad_w: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_horizontal_advection_term_for_vertical_velocity(
        vn_ie=vn_ie,
        inv_dual_edge_length=inv_dual_edge_length,
        w=w,
        z_vt_ie=z_vt_ie,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        z_w_v=z_w_v,
        out=z_v_grad_w,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

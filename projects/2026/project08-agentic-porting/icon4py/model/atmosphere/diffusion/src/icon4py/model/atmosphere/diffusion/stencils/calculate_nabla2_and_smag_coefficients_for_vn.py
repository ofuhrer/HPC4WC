# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, maximum, minimum, sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C2V, E2C2VDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _calculate_nabla2_and_smag_coefficients_for_vn(
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
    smag_limit: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    smag_offset: vpfloat,
) -> tuple[
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[wpfloat],
]:
    diff_multfac_smag_wp, u_vert_wp, v_vert_wp, smag_offset_wp = astype(
        (diff_multfac_smag, u_vert, v_vert, smag_offset), wpfloat
    )

    v_n = u_vert_wp(E2C2V) * primal_normal_vert_x + v_vert_wp(E2C2V) * primal_normal_vert_y

    nabla2_of_vn = (v_n[E2C2VDim(0)] + v_n[E2C2VDim(1)] - wpfloat("2.0") * vn) * (
        inv_primal_edge_length * inv_primal_edge_length
    ) + (v_n[E2C2VDim(2)] + v_n[E2C2VDim(3)] - wpfloat("2.0") * vn) * (
        inv_vert_vert_length * inv_vert_vert_length
    )
    # The factor of 4 comes from the lengths in the denominator being twice those needed
    # for the diffusion stencil (https://doi.org/10.1002%2Fqj.2378).
    nabla2_of_vn = wpfloat("4.0") * nabla2_of_vn

    v_t = u_vert_wp(E2C2V) * dual_normal_vert_x + v_vert_wp(E2C2V) * dual_normal_vert_y
    l_p = tangent_orientation * inv_primal_edge_length
    l_vv = inv_vert_vert_length

    kh_smag_part1 = (v_n[E2C2VDim(3)] - v_n[E2C2VDim(2)]) * l_vv - (
        v_t[E2C2VDim(1)] - v_t[E2C2VDim(0)]
    ) * l_p
    kh_smag_part2 = (v_n[E2C2VDim(1)] - v_n[E2C2VDim(0)]) * l_p + (
        v_t[E2C2VDim(3)] - v_t[E2C2VDim(2)]
    ) * l_vv

    kh_smag_wp = diff_multfac_smag_wp * sqrt(
        kh_smag_part1 * kh_smag_part1 + kh_smag_part2 * kh_smag_part2
    )

    kh_smag_vp_for_diffusion = maximum(vpfloat("0.0"), astype(kh_smag_wp - smag_offset_wp, vpfloat))
    kh_smag_vp_for_diffusion = minimum(kh_smag_vp_for_diffusion, smag_limit)

    return kh_smag_vp_for_diffusion, astype(kh_smag_wp, vpfloat), nabla2_of_vn


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_nabla2_and_smag_coefficients_for_vn(
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
    smag_limit: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    kh_smag_ec: fa.EdgeKField[vpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    smag_offset: vpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _calculate_nabla2_and_smag_coefficients_for_vn(
        diff_multfac_smag=diff_multfac_smag,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        u_vert=u_vert,
        v_vert=v_vert,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        vn=vn,
        smag_limit=smag_limit,
        smag_offset=smag_offset,
        out=(kh_smag_e, kh_smag_ec, z_nabla2_e),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

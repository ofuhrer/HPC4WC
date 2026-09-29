# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import astype, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2CDim
from icon4py.model.common.type_alias import vpfloat


@gtx.field_operator
def _compute_barycentric_backtrajectory(
    p_vn: fa.EdgeKField[ta.wpfloat],
    p_vt: fa.EdgeKField[ta.wpfloat],
    cell_idx: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.int32],
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    p_dthalf: ta.wpfloat,
) -> tuple[
    fa.EdgeKField[gtx.int32],
    fa.EdgeKField[gtx.int32],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    lvn_pos = p_vn >= 0.0

    p_cell_idx = where(lvn_pos, cell_idx[E2CDim(0)], cell_idx[E2CDim(1)])
    p_cell_rel_idx_dsl = where(lvn_pos, 0, 1)

    z_ntdistv_bary_1 = -(
        p_vn * p_dthalf + where(lvn_pos, pos_on_tplane_e_1[E2CDim(0)], pos_on_tplane_e_1[E2CDim(1)])
    )

    z_ntdistv_bary_2 = -(
        p_vt * p_dthalf + where(lvn_pos, pos_on_tplane_e_2[E2CDim(0)], pos_on_tplane_e_2[E2CDim(1)])
    )

    p_distv_bary_1 = where(
        lvn_pos,
        z_ntdistv_bary_1 * primal_normal_cell_1[E2CDim(0)]
        + z_ntdistv_bary_2 * dual_normal_cell_1[E2CDim(0)],
        z_ntdistv_bary_1 * primal_normal_cell_1[E2CDim(1)]
        + z_ntdistv_bary_2 * dual_normal_cell_1[E2CDim(1)],
    )

    p_distv_bary_2 = where(
        lvn_pos,
        z_ntdistv_bary_1 * primal_normal_cell_2[E2CDim(0)]
        + z_ntdistv_bary_2 * dual_normal_cell_2[E2CDim(0)],
        z_ntdistv_bary_1 * primal_normal_cell_2[E2CDim(1)]
        + z_ntdistv_bary_2 * dual_normal_cell_2[E2CDim(1)],
    )

    return (
        p_cell_idx,
        p_cell_rel_idx_dsl,
        astype(p_distv_bary_1, vpfloat),
        astype(p_distv_bary_2, vpfloat),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_barycentric_backtrajectory(
    p_vn: fa.EdgeKField[ta.wpfloat],
    p_vt: fa.EdgeKField[ta.wpfloat],
    cell_idx: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.int32],
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    p_cell_idx: fa.EdgeKField[gtx.int32],
    p_cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
    p_distv_bary_1: fa.EdgeKField[ta.vpfloat],
    p_distv_bary_2: fa.EdgeKField[ta.vpfloat],
    p_dthalf: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_barycentric_backtrajectory(
        p_vn=p_vn,
        p_vt=p_vt,
        cell_idx=cell_idx,
        pos_on_tplane_e_1=pos_on_tplane_e_1,
        pos_on_tplane_e_2=pos_on_tplane_e_2,
        primal_normal_cell_1=primal_normal_cell_1,
        dual_normal_cell_1=dual_normal_cell_1,
        primal_normal_cell_2=primal_normal_cell_2,
        dual_normal_cell_2=dual_normal_cell_2,
        p_dthalf=p_dthalf,
        out=(
            p_cell_idx,
            p_cell_rel_idx_dsl,
            p_distv_bary_1,
            p_distv_bary_2,
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

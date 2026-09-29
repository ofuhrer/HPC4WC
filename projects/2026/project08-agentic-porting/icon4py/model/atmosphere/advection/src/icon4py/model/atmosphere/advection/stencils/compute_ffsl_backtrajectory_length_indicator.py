# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import sqrt, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2CDim


@gtx.field_operator
def _compute_ffsl_backtrajectory_length_indicator(
    p_vn: fa.EdgeKField[ta.wpfloat],
    p_vt: fa.EdgeKField[ta.wpfloat],
    edge_cell_length: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    p_dt: ta.wpfloat,
) -> fa.EdgeKField[gtx.int32]:
    traj_length = sqrt(p_vn * p_vn + p_vt * p_vt) * p_dt
    e2c_length = where(p_vn >= 0.0, edge_cell_length[E2CDim(0)], edge_cell_length[E2CDim(1)])
    opt_famask_dsl = where(traj_length > 1.25 * e2c_length, 1, 0)
    return opt_famask_dsl


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ffsl_backtrajectory_length_indicator(
    p_vn: fa.EdgeKField[ta.wpfloat],
    p_vt: fa.EdgeKField[ta.wpfloat],
    edge_cell_length: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    opt_famask_dsl: fa.EdgeKField[gtx.int32],
    p_dt: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ffsl_backtrajectory_length_indicator(
        p_vn=p_vn,
        p_vt=p_vt,
        edge_cell_length=edge_cell_length,
        p_dt=p_dt,
        out=opt_famask_dsl,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

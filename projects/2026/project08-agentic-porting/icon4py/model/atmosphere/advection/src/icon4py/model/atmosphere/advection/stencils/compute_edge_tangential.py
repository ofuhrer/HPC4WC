# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C2E


# TODO(dastrm): this stencil is a duplicate of compute_tangential_wind
# TODO(dastrm): this stencil has no test


@gtx.field_operator
def _compute_edge_tangential(
    p_vn_in: fa.EdgeKField[ta.wpfloat],
    ptr_coeff: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    p_vt_out = neighbor_sum(p_vn_in(E2C2E) * ptr_coeff, axis=dims.E2C2EDim)
    return p_vt_out


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_edge_tangential(
    p_vn_in: fa.EdgeKField[ta.wpfloat],
    ptr_coeff: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    p_vt_out: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_edge_tangential(
        p_vn_in=p_vn_in,
        ptr_coeff=ptr_coeff,
        out=p_vt_out,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

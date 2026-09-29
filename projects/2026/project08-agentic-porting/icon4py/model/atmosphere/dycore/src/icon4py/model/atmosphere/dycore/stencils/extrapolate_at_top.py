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
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _extrapolate_at_top(
    wgtfacq_e: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly known as mo_velocity_advection_stencil_06 or mo_solve_nonhydro_stencil_38."""
    wgtfacq_e_wp = astype(wgtfacq_e, wpfloat)

    vn_ie_wp = (
        wgtfacq_e_wp(KDim - 1) * vn(KDim - 1)
        + wgtfacq_e_wp(KDim - 2) * vn(KDim - 2)
        + wgtfacq_e_wp(KDim - 3) * vn(KDim - 3)
    )

    return astype(vn_ie_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def extrapolate_at_top(
    wgtfacq_e: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    vn_ie: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _extrapolate_at_top(
        wgtfacq_e=wgtfacq_e,
        vn=vn,
        out=vn_ie,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

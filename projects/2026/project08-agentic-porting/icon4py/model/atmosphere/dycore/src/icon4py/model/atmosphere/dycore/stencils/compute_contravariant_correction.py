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
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_contravariant_correction(
    vn: fa.EdgeKField[wpfloat],
    ddxn_z_full: fa.EdgeKField[vpfloat],
    ddxt_z_full: fa.EdgeKField[vpfloat],
    vt: fa.EdgeKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_35 or mo_velocity_advection_stencil_04."""
    ddxn_z_full_wp = astype(ddxn_z_full, wpfloat)

    z_w_concorr_me_wp = vn * ddxn_z_full_wp + astype(vt * ddxt_z_full, wpfloat)
    return astype(z_w_concorr_me_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_contravariant_correction(
    vn: fa.EdgeKField[wpfloat],
    ddxn_z_full: fa.EdgeKField[vpfloat],
    ddxt_z_full: fa.EdgeKField[vpfloat],
    vt: fa.EdgeKField[vpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_contravariant_correction(
        vn=vn,
        ddxn_z_full=ddxn_z_full,
        ddxt_z_full=ddxt_z_full,
        vt=vt,
        out=z_w_concorr_me,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

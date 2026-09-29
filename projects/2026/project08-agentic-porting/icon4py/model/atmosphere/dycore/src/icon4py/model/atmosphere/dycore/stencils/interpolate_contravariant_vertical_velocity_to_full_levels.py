# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import vpfloat


@gtx.field_operator
def _interpolate_contravariant_vertical_velocity_to_full_levels(
    z_w_con_c: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly know as _mo_velocity_advection_stencil_15."""
    z_w_con_c_full_vp = vpfloat("0.5") * (z_w_con_c + z_w_con_c(KDim + 1))
    return z_w_con_c_full_vp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_contravariant_vertical_velocity_to_full_levels(
    z_w_con_c: fa.CellKField[vpfloat],
    z_w_con_c_full: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    # TODO(): this should use a generic stencil
    _interpolate_contravariant_vertical_velocity_to_full_levels(
        z_w_con_c=z_w_con_c,
        out=z_w_con_c_full,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

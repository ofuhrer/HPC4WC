# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.atmosphere.dycore.dycore_utils import (
    _calculate_fourth_order_divdamp_scaling_coeff,
)
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _apply_4th_order_divergence_damping(
    interpolated_fourth_order_divdamp_factor: fa.KField[wpfloat],
    z_graddiv2_vn: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    divdamp_order: gtx.int32,
    mean_cell_area: float,
    second_order_divdamp_factor: float,
) -> fa.EdgeKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_4th_order_divdamp."""
    scal_divdamp = _calculate_fourth_order_divdamp_scaling_coeff(
        interpolated_fourth_order_divdamp_factor,
        divdamp_order,
        mean_cell_area,
        second_order_divdamp_factor,
    )
    z_graddiv2_vn_wp = astype(z_graddiv2_vn, wpfloat)
    vn_wp = vn + (scal_divdamp * z_graddiv2_vn_wp)
    return vn_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_4th_order_divergence_damping(
    interpolated_fourth_order_divdamp_factor: fa.KField[wpfloat],
    z_graddiv2_vn: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    divdamp_order: gtx.int32,
    mean_cell_area: float,
    second_order_divdamp_factor: float,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_4th_order_divergence_damping(
        interpolated_fourth_order_divdamp_factor=interpolated_fourth_order_divdamp_factor,
        z_graddiv2_vn=z_graddiv2_vn,
        vn=vn,
        divdamp_order=divdamp_order,
        mean_cell_area=mean_cell_area,
        second_order_divdamp_factor=second_order_divdamp_factor,
        out=vn,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, neighbor_sum, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E2C, C2E2CDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _temporary_field_for_grid_point_cold_pools_enhancement(
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    thresh_tdiff: wpfloat,
    smallest_vpfloat: vpfloat,
) -> fa.CellKField[vpfloat]:
    theta_ref_mc_wp = astype(theta_ref_mc, wpfloat)

    tdiff = theta_v - neighbor_sum(theta_v(C2E2C), axis=C2E2CDim) / wpfloat("3.0")
    trefdiff = theta_ref_mc_wp - astype(
        neighbor_sum(theta_ref_mc(C2E2C), axis=C2E2CDim), wpfloat
    ) / wpfloat("3.0")
    enh_diffu_3d_vp = where(
        ((tdiff - trefdiff) < thresh_tdiff) & (trefdiff < wpfloat("0.0"))
        | (tdiff - trefdiff < wpfloat("1.5") * thresh_tdiff),
        astype((thresh_tdiff - tdiff + trefdiff) * wpfloat("5.0e-4"), vpfloat),
        smallest_vpfloat,
    )
    return enh_diffu_3d_vp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def temporary_field_for_grid_point_cold_pools_enhancement(
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    enh_diffu_3d: fa.CellKField[vpfloat],
    thresh_tdiff: wpfloat,
    smallest_vpfloat: vpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _temporary_field_for_grid_point_cold_pools_enhancement(
        theta_v=theta_v,
        theta_ref_mc=theta_ref_mc,
        thresh_tdiff=thresh_tdiff,
        smallest_vpfloat=smallest_vpfloat,
        out=enh_diffu_3d,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

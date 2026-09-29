# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.atmosphere.diffusion.stencils.enhance_diffusion_coefficient_for_grid_point_cold_pools import (
    _enhance_diffusion_coefficient_for_grid_point_cold_pools,
)
from icon4py.model.atmosphere.diffusion.stencils.temporary_field_for_grid_point_cold_pools_enhancement import (
    _temporary_field_for_grid_point_cold_pools_enhancement,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    thresh_tdiff: wpfloat,
    smallest_vpfloat: vpfloat,
    kh_smag_e: fa.EdgeKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    enh_diffu_3d = _temporary_field_for_grid_point_cold_pools_enhancement(
        theta_v,
        theta_ref_mc,
        thresh_tdiff,
        smallest_vpfloat,
    )
    kh_smag_e_vp = _enhance_diffusion_coefficient_for_grid_point_cold_pools(kh_smag_e, enh_diffu_3d)
    return kh_smag_e_vp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    thresh_tdiff: wpfloat,
    smallest_vpfloat: vpfloat,
    kh_smag_e: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
        theta_v=theta_v,
        theta_ref_mc=theta_ref_mc,
        thresh_tdiff=thresh_tdiff,
        smallest_vpfloat=smallest_vpfloat,
        kh_smag_e=kh_smag_e,
        out=kh_smag_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

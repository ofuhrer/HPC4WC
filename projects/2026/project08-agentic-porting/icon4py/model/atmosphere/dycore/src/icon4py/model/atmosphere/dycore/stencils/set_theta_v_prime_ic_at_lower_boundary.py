# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.atmosphere.dycore.stencils.interpolate_to_surface import _interpolate_to_surface
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _set_theta_v_prime_ic_at_lower_boundary(
    wgtfacq_c: fa.CellKField[vpfloat],
    z_rth_pr: fa.CellKField[vpfloat],
    theta_ref_ic: fa.CellKField[vpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_11_upper."""
    z_theta_v_pr_ic_vp = _interpolate_to_surface(wgtfacq_c=wgtfacq_c, interpolant=z_rth_pr)
    theta_v_ic_vp = theta_ref_ic + z_theta_v_pr_ic_vp
    return z_theta_v_pr_ic_vp, astype(theta_v_ic_vp, wpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def set_theta_v_prime_ic_at_lower_boundary(
    wgtfacq_c: fa.CellKField[vpfloat],
    z_rth_pr: fa.CellKField[vpfloat],
    theta_ref_ic: fa.CellKField[vpfloat],
    z_theta_v_pr_ic: fa.CellKField[vpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _set_theta_v_prime_ic_at_lower_boundary(
        wgtfacq_c=wgtfacq_c,
        z_rth_pr=z_rth_pr,
        theta_ref_ic=theta_ref_ic,
        out=(z_theta_v_pr_ic, theta_v_ic),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

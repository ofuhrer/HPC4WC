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
def _compute_explicit_vertical_wind_from_advection_and_vertical_wind_density(
    w_nnow: fa.CellKField[wpfloat],
    ddt_w_adv_ntl1: fa.CellKField[vpfloat],
    ddt_w_adv_ntl2: fa.CellKField[vpfloat],
    z_th_ddz_exner_c: fa.CellKField[vpfloat],
    rho_ic: fa.CellKField[wpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    vwind_expl_wgt: fa.CellField[wpfloat],
    dtime: wpfloat,
    wgt_nnow_vel: wpfloat,
    wgt_nnew_vel: wpfloat,
    cpd: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_42."""
    ddt_w_adv_ntl1_wp, ddt_w_adv_ntl2_wp, z_th_ddz_exner_c_wp, w_concorr_c_wp = astype(
        (ddt_w_adv_ntl1, ddt_w_adv_ntl2, z_th_ddz_exner_c, w_concorr_c), wpfloat
    )

    z_w_expl_wp = w_nnow + dtime * (
        wgt_nnow_vel * ddt_w_adv_ntl1_wp
        + wgt_nnew_vel * ddt_w_adv_ntl2_wp
        - cpd * z_th_ddz_exner_c_wp
    )
    z_contr_w_fl_l_wp = rho_ic * (-w_concorr_c_wp + vwind_expl_wgt * w_nnow)
    return z_w_expl_wp, z_contr_w_fl_l_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_explicit_vertical_wind_from_advection_and_vertical_wind_density(
    z_w_expl: fa.CellKField[wpfloat],
    w_nnow: fa.CellKField[wpfloat],
    ddt_w_adv_ntl1: fa.CellKField[vpfloat],
    ddt_w_adv_ntl2: fa.CellKField[vpfloat],
    z_th_ddz_exner_c: fa.CellKField[vpfloat],
    z_contr_w_fl_l: fa.CellKField[wpfloat],
    rho_ic: fa.CellKField[wpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    vwind_expl_wgt: fa.CellField[wpfloat],
    dtime: wpfloat,
    wgt_nnow_vel: wpfloat,
    wgt_nnew_vel: wpfloat,
    cpd: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_explicit_vertical_wind_from_advection_and_vertical_wind_density(
        w_nnow=w_nnow,
        ddt_w_adv_ntl1=ddt_w_adv_ntl1,
        ddt_w_adv_ntl2=ddt_w_adv_ntl2,
        z_th_ddz_exner_c=z_th_ddz_exner_c,
        rho_ic=rho_ic,
        w_concorr_c=w_concorr_c,
        vwind_expl_wgt=vwind_expl_wgt,
        dtime=dtime,
        wgt_nnow_vel=wgt_nnow_vel,
        wgt_nnew_vel=wgt_nnew_vel,
        cpd=cpd,
        out=(z_w_expl, z_contr_w_fl_l),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

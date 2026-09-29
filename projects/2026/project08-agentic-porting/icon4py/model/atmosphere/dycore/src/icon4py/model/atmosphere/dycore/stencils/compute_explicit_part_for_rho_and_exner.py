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
def _compute_explicit_part_for_rho_and_exner(
    rho_nnow: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    z_flxdiv_mass: fa.CellKField[vpfloat],
    z_contr_w_fl_l: fa.CellKField[wpfloat],
    exner_pr: fa.CellKField[wpfloat],
    z_beta: fa.CellKField[vpfloat],
    z_flxdiv_theta: fa.CellKField[vpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    ddt_exner_phy: fa.CellKField[vpfloat],
    dtime: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_48 or _mo_solve_nonhydro_stencil_49."""
    inv_ddqz_z_full_wp, z_flxdiv_mass_wp, z_beta_wp, z_flxdiv_theta_wp, ddt_exner_phy_wp = astype(
        (inv_ddqz_z_full, z_flxdiv_mass, z_beta, z_flxdiv_theta, ddt_exner_phy), wpfloat
    )

    z_rho_expl_wp = rho_nnow - dtime * inv_ddqz_z_full_wp * (
        z_flxdiv_mass_wp + z_contr_w_fl_l - z_contr_w_fl_l(KDim + 1)
    )

    z_exner_expl_wp = (
        exner_pr
        - z_beta_wp
        * (
            z_flxdiv_theta_wp
            + theta_v_ic * z_contr_w_fl_l
            - theta_v_ic(KDim + 1) * z_contr_w_fl_l(KDim + 1)
        )
        + dtime * ddt_exner_phy_wp
    )
    return z_rho_expl_wp, z_exner_expl_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_explicit_part_for_rho_and_exner(
    z_rho_expl: fa.CellKField[wpfloat],
    z_exner_expl: fa.CellKField[wpfloat],
    rho_nnow: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    z_flxdiv_mass: fa.CellKField[vpfloat],
    z_contr_w_fl_l: fa.CellKField[wpfloat],
    exner_pr: fa.CellKField[wpfloat],
    z_beta: fa.CellKField[vpfloat],
    z_flxdiv_theta: fa.CellKField[vpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    ddt_exner_phy: fa.CellKField[vpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_explicit_part_for_rho_and_exner(
        rho_nnow=rho_nnow,
        inv_ddqz_z_full=inv_ddqz_z_full,
        z_flxdiv_mass=z_flxdiv_mass,
        z_contr_w_fl_l=z_contr_w_fl_l,
        exner_pr=exner_pr,
        z_beta=z_beta,
        z_flxdiv_theta=z_flxdiv_theta,
        theta_v_ic=theta_v_ic,
        ddt_exner_phy=ddt_exner_phy,
        dtime=dtime,
        out=(z_rho_expl, z_exner_expl),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

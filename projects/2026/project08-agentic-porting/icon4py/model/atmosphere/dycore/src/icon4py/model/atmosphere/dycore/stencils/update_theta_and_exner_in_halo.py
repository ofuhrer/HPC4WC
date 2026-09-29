# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import exp, log, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _update_theta_and_exner_in_halo(
    mask_prog_halo_c: fa.CellField[bool],
    rho_now: fa.CellKField[wpfloat],
    rho_new: fa.CellKField[wpfloat],
    theta_v_now: fa.CellKField[wpfloat],
    exner_now: fa.CellKField[wpfloat],
    exner_new: fa.CellKField[wpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    #
    # ICON (fortran) uses one list (bdy_halo_c) and one mask (mask_prog_halo_c)
    # for these operations. In the halo one is the inverse of the other:
    # `mask_prog_halo_c = ~bdy_halo_c` so these operations (formerly
    # _mo_solve_nonhydro_stencil_66 and 68) can be merged.
    # `mask_prog_halo_c` is true for prognostic (_prog) cells (_c) in the halo
    # region (_halo)
    # `bdy_halo_c` lists the indices of cells (_c) which are located in the
    # lateral boundary (bdy) and halo region (_halo)
    # A visual representation can be found at
    # https://github.com/C2SM/icon4py/pull/1066

    theta_v_new = where(
        mask_prog_halo_c,
        rho_now
        * theta_v_now
        * ((exner_new / exner_now - wpfloat("1.0")) * PhysicsConstants.cvd_o_rd + wpfloat("1.0"))
        / rho_new,
        exner_new,
    )
    exner_new = where(
        mask_prog_halo_c,
        exner_new,
        exp(PhysicsConstants.rd_o_cvd * log(PhysicsConstants.rd_o_p0ref * rho_new * exner_new)),
    )

    return theta_v_new, exner_new


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_theta_and_exner_in_halo(
    mask_prog_halo_c: fa.CellField[bool],
    rho_now: fa.CellKField[wpfloat],
    rho_new: fa.CellKField[wpfloat],
    theta_v_now: fa.CellKField[wpfloat],
    theta_v_new: fa.CellKField[wpfloat],
    exner_new: fa.CellKField[wpfloat],
    exner_now: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_theta_and_exner_in_halo(
        mask_prog_halo_c=mask_prog_halo_c,
        rho_now=rho_now,
        rho_new=rho_new,
        theta_v_now=theta_v_now,
        exner_now=exner_now,
        exner_new=exner_new,
        out=(theta_v_new, exner_new),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

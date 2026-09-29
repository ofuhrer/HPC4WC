# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E, C2EDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_divergence_of_fluxes_of_rho_and_theta(
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    mass_fl_e: fa.EdgeKField[wpfloat],
    z_theta_v_fl_e: fa.EdgeKField[wpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_41."""
    z_flxdiv_mass_wp = neighbor_sum(geofac_div * mass_fl_e(C2E), axis=C2EDim)
    z_flxdiv_theta_wp = neighbor_sum(geofac_div * z_theta_v_fl_e(C2E), axis=C2EDim)
    return astype((z_flxdiv_mass_wp, z_flxdiv_theta_wp), vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_divergence_of_fluxes_of_rho_and_theta(
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    mass_fl_e: fa.EdgeKField[wpfloat],
    z_theta_v_fl_e: fa.EdgeKField[wpfloat],
    z_flxdiv_mass: fa.CellKField[vpfloat],
    z_flxdiv_theta: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_divergence_of_fluxes_of_rho_and_theta(
        geofac_div=geofac_div,
        mass_fl_e=mass_fl_e,
        z_theta_v_fl_e=z_theta_v_fl_e,
        out=(z_flxdiv_mass, z_flxdiv_theta),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

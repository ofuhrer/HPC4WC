# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C2EO, E2C2EODim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _spatially_average_flux_or_velocity(
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, E2C2EODim], wpfloat],
    flux_or_velocity: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_31."""
    # e_flx_avg is the weight vector for spatially averaging fluxes or velocities
    spatially_averaged_flux_or_velocity = neighbor_sum(
        e_flx_avg * flux_or_velocity(E2C2EO), axis=E2C2EODim
    )
    return spatially_averaged_flux_or_velocity


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def spatially_average_flux_or_velocity(
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, E2C2EODim], wpfloat],
    flux_or_velocity: fa.EdgeKField[wpfloat],
    spatially_averaged_flux_or_velocity: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _spatially_average_flux_or_velocity(
        e_flx_avg=e_flx_avg,
        flux_or_velocity=flux_or_velocity,
        out=spatially_averaged_flux_or_velocity,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

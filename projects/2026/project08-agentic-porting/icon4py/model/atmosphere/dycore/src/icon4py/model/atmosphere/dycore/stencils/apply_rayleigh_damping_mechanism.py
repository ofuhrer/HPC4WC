# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _apply_rayleigh_damping_mechanism(
    z_raylfac: fa.KField[wpfloat],
    w: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_54."""
    z_raylfac = broadcast(z_raylfac, (dims.CellDim, dims.KDim))
    w_wp = z_raylfac * w
    return w_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_rayleigh_damping_mechanism(
    z_raylfac: fa.KField[wpfloat],
    w: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_rayleigh_damping_mechanism(
        z_raylfac=z_raylfac,
        w=w,
        out=w,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

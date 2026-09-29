# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import exp, log

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _compute_exner_from_rhotheta(
    rho: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    rd_o_cvd: wpfloat,
    rd_o_p0ref: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_67."""
    theta_v_wp = exner
    exner_wp = exp(rd_o_cvd * log(rd_o_p0ref * rho * theta_v_wp))
    return theta_v_wp, exner_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_exner_from_rhotheta(
    rho: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    rd_o_cvd: wpfloat,
    rd_o_p0ref: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_exner_from_rhotheta(
        rho=rho,
        exner=exner,
        rd_o_cvd=rd_o_cvd,
        rd_o_p0ref=rd_o_p0ref,
        out=(theta_v, exner),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

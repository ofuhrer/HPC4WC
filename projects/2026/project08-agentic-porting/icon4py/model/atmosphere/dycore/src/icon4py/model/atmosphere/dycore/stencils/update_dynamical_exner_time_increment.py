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
def _update_dynamical_exner_time_increment(
    exner: fa.CellKField[wpfloat],
    ddt_exner_phy: fa.CellKField[vpfloat],
    exner_dyn_incr: fa.CellKField[vpfloat],
    ndyn_substeps_var: wpfloat,
    dtime: wpfloat,
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_60."""
    exner_dyn_incr_wp, ddt_exner_phy_wp = astype((exner_dyn_incr, ddt_exner_phy), wpfloat)

    exner_dyn_incr_wp = exner - (exner_dyn_incr_wp + ndyn_substeps_var * dtime * ddt_exner_phy_wp)
    return astype(exner_dyn_incr_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_dynamical_exner_time_increment(
    exner: fa.CellKField[wpfloat],
    ddt_exner_phy: fa.CellKField[vpfloat],
    exner_dyn_incr: fa.CellKField[vpfloat],
    ndyn_substeps_var: wpfloat,
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_dynamical_exner_time_increment(
        exner=exner,
        ddt_exner_phy=ddt_exner_phy,
        exner_dyn_incr=exner_dyn_incr,
        ndyn_substeps_var=ndyn_substeps_var,
        dtime=dtime,
        out=exner_dyn_incr,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

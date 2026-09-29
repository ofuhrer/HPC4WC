# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


# TODO(dastrm): move this highly generic stencil to common
# TODO(dastrm): this stencil has no test


@gtx.field_operator
def _copy_cell_kdim_field(field_in: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    return field_in


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def copy_cell_kdim_field(
    field_in: fa.CellKField[ta.wpfloat],
    field_out: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _copy_cell_kdim_field(
        field_in=field_in,
        out=field_out,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

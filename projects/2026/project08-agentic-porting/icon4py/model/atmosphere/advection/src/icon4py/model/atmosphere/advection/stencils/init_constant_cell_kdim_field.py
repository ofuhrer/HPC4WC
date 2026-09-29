# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


# TODO(dastrm): move this highly generic stencil to common
# TODO(dastrm): this stencil has no test


@gtx.field_operator
def _init_constant_cell_kdim_field(value: ta.wpfloat) -> fa.CellKField[ta.wpfloat]:
    return broadcast(value, (dims.CellDim, dims.KDim))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def init_constant_cell_kdim_field(
    field: fa.CellKField[ta.wpfloat],
    value: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _init_constant_cell_kdim_field(
        value=value,
        out=field,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

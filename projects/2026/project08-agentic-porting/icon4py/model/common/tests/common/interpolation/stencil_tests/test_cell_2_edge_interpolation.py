# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    cell_2_edge_interpolation,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


@pytest.mark.skip_value_error
class TestCell2EdgeInterpolation(stencil_tests.StencilTest):
    PROGRAM = cell_2_edge_interpolation
    OUTPUTS = ("out_field",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        in_field: np.ndarray,
        coeff: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        e2c = connectivities[dims.E2CDim]
        coeff_ = np.expand_dims(coeff, axis=-1)
        out_field = np.sum(in_field[e2c] * coeff_, axis=1)

        return dict(
            out_field=out_field,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        in_field = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        coeff = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat)
        out_field = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)

        return dict(
            in_field=in_field,
            coeff=coeff,
            out_field=out_field,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )

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

from icon4py.model.atmosphere.advection.stencils.compute_vertical_parabola_limiter_condition import (
    compute_vertical_parabola_limiter_condition,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestComputeVerticalParabolaLimiterCondition(stencil_tests.StencilTest):
    PROGRAM = compute_vertical_parabola_limiter_condition
    OUTPUTS = ("l_limit",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_face: np.ndarray,
        p_cc: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        z_delta = p_face[:, :-1] - p_face[:, 1:]
        z_a6i = 6.0 * (p_cc - 0.5 * (p_face[:, :-1] + p_face[:, 1:]))
        l_limit = np.where(np.abs(z_delta) < -1 * z_a6i, 1, 0)
        return dict(l_limit=l_limit)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        p_cc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_face = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        l_limit = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=gtx.int32)
        return dict(
            p_face=p_face,
            p_cc=p_cc,
            l_limit=l_limit,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

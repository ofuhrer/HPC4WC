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

from icon4py.model.atmosphere.advection.stencils.compute_tendency import compute_tendency
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestComputeTendency(stencil_tests.StencilTest):
    PROGRAM = compute_tendency
    OUTPUTS = ("opt_ddt_tracer_adv",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_tracer_now: np.ndarray,
        p_tracer_new: np.ndarray,
        p_dtime,
        **kwargs: Any,
    ) -> dict:
        opt_ddt_tracer_adv = (p_tracer_new - p_tracer_now) / p_dtime

        return dict(opt_ddt_tracer_adv=opt_ddt_tracer_adv)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        p_tracer_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_tracer_new = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        opt_ddt_tracer_adv = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_dtime = np.float64(5.0)
        return dict(
            p_tracer_now=p_tracer_now,
            p_tracer_new=p_tracer_new,
            p_dtime=p_dtime,
            opt_ddt_tracer_adv=opt_ddt_tracer_adv,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

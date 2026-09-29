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

from icon4py.model.atmosphere.advection.stencils.average_horizontal_flux_subcycling_3 import (
    average_horizontal_flux_subcycling_3,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestAverageHorizontalFluxSubcycling3(stencil_tests.StencilTest):
    PROGRAM = average_horizontal_flux_subcycling_3
    OUTPUTS = ("p_out_e",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        z_tracer_mflx_1_dsl: np.ndarray,
        z_tracer_mflx_2_dsl: np.ndarray,
        z_tracer_mflx_3_dsl: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        p_out_e = (z_tracer_mflx_1_dsl + z_tracer_mflx_2_dsl + z_tracer_mflx_3_dsl) / float(3)
        return dict(p_out_e=p_out_e)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        z_tracer_mflx_1_dsl = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_tracer_mflx_2_dsl = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_tracer_mflx_3_dsl = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_out_e = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        return dict(
            z_tracer_mflx_1_dsl=z_tracer_mflx_1_dsl,
            z_tracer_mflx_2_dsl=z_tracer_mflx_2_dsl,
            z_tracer_mflx_3_dsl=z_tracer_mflx_3_dsl,
            p_out_e=p_out_e,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

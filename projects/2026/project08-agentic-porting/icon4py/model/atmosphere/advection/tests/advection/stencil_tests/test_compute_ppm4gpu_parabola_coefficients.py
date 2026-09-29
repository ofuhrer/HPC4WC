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

from icon4py.model.atmosphere.advection.stencils.compute_ppm4gpu_parabola_coefficients import (
    compute_ppm4gpu_parabola_coefficients,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestComputePpm4gpuParabolaCoefficients(stencil_tests.StencilTest):
    PROGRAM = compute_ppm4gpu_parabola_coefficients
    OUTPUTS = ("z_delta_q", "z_a1")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        z_face_up: np.ndarray,
        z_face_low: np.ndarray,
        p_cc: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        z_delta_q = 0.5 * (z_face_up - z_face_low)
        z_a1 = p_cc - 0.5 * (z_face_up + z_face_low)
        return dict(z_delta_q=z_delta_q, z_a1=z_a1)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        z_face_up = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_face_low = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_cc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_delta_q = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        z_a1 = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        return dict(
            z_face_up=z_face_up,
            z_face_low=z_face_low,
            p_cc=p_cc,
            z_delta_q=z_delta_q,
            z_a1=z_a1,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

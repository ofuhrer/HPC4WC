# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import cloud_x_ice
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestCloudXIceDefault(StencilTest):
    PROGRAM = cloud_x_ice
    OUTPUTS = ("freezing_rate",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        t: np.ndarray,
        qc: np.ndarray,
        qi: np.ndarray,
        dt: wpfloat,
        **kwargs,
    ) -> dict:
        return dict(freezing_rate=np.full(t.shape, 0.0))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t=data_alloc.constant_field(grid, 256.835, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            qi=data_alloc.constant_field(grid, 4.50245e-7, dims.CellDim, dims.KDim, dtype=wpfloat),
            dt=30.0,
            freezing_rate=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )

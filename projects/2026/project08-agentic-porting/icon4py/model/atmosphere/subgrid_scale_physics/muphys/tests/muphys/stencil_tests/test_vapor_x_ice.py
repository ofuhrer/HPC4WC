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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import vapor_x_ice
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestVaporXIceDefault(StencilTest):
    PROGRAM = vapor_x_ice
    OUTPUTS = ("vapor_deposition_rate",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        qi: np.ndarray,
        mi: np.ndarray,
        eta: np.ndarray,
        dvsi: np.ndarray,
        rho: np.ndarray,
        dt: wpfloat,
        **kwargs,
    ) -> dict:
        return dict(vapor_deposition_rate=np.full(qi.shape, 2.2106162342610385e-09))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            qi=data_alloc.constant_field(grid, 9.53048e-07, dims.CellDim, dims.KDim, dtype=wpfloat),
            mi=data_alloc.constant_field(grid, 1.0e-9, dims.CellDim, dims.KDim, dtype=wpfloat),
            eta=data_alloc.constant_field(
                grid, 1.90278e-05, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            dvsi=data_alloc.constant_field(
                grid, 0.000120375, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            rho=data_alloc.constant_field(grid, 1.19691, dims.CellDim, dims.KDim, dtype=wpfloat),
            dt=30.0,
            vapor_deposition_rate=data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
        )

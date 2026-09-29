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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import (
    vel_scale_factor_snow,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestVelScaleFactorSnow(StencilTest):
    PROGRAM = vel_scale_factor_snow
    OUTPUTS = ("scale_factor",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        xrho: np.ndarray,
        rho: np.ndarray,
        t: np.ndarray,
        qs: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(scale_factor=np.full(xrho.shape, 0.06633230453931642))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            xrho=data_alloc.constant_field(grid, 1.17787, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=data_alloc.constant_field(grid, 0.882961, dims.CellDim, dims.KDim, dtype=wpfloat),
            t=data_alloc.constant_field(grid, 257.101, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs=data_alloc.constant_field(grid, 5.78761e-06, dims.CellDim, dims.KDim, dtype=wpfloat),
            scale_factor=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )

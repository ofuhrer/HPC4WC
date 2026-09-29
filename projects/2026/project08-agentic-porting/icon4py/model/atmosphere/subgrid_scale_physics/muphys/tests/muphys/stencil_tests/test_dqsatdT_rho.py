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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import dqsatdT_rho
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestQsatRho(StencilTest):
    PROGRAM = dqsatdT_rho
    OUTPUTS = ("derivative",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        qs: np.ndarray,
        t: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(derivative=np.full(t.shape, 0.00030825070286492049))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            qs=data_alloc.constant_field(grid, 0.00448941, dims.CellDim, dims.KDim, dtype=wpfloat),
            t=data_alloc.constant_field(grid, 273.909, dims.CellDim, dims.KDim, dtype=wpfloat),
            derivative=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )

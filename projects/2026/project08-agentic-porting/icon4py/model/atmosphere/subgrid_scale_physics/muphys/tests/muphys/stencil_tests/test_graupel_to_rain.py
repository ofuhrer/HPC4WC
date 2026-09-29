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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import graupel_to_rain
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestGraupelToRain(StencilTest):
    PROGRAM = graupel_to_rain
    OUTPUTS = ("rain_rate",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        t: np.ndarray,
        p: np.ndarray,
        rho: np.ndarray,
        dvsw0: np.ndarray,
        qg: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(rain_rate=np.full(t.shape, 5.9748142538569357e-13))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t=data_alloc.constant_field(grid, 280.156, dims.CellDim, dims.KDim, dtype=wpfloat),
            p=data_alloc.constant_field(grid, 98889.4, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=data_alloc.constant_field(grid, 1.22804, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw0=data_alloc.constant_field(
                grid, -0.00167867, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            qg=data_alloc.constant_field(grid, 1.53968e-15, dims.CellDim, dims.KDim, dtype=wpfloat),
            rain_rate=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )

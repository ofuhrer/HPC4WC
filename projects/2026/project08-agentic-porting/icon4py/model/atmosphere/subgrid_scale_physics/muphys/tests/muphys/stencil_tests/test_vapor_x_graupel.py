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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import vapor_x_graupel
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestVaporXGraupel(StencilTest):
    PROGRAM = vapor_x_graupel
    OUTPUTS = ("exchange_rate",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        t: np.ndarray,
        p: np.ndarray,
        rho: np.ndarray,
        qg: np.ndarray,
        dvsw: np.ndarray,
        dvsi: np.ndarray,
        dvsw0: np.ndarray,
        dt: wpfloat,
        **kwargs,
    ) -> dict:
        return dict(exchange_rate=np.full(t.shape, -6.8498666666666675e-13))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t=data_alloc.constant_field(grid, 278.026, dims.CellDim, dims.KDim, dtype=wpfloat),
            p=data_alloc.constant_field(grid, 95987.1, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=data_alloc.constant_field(grid, 1.20041, dims.CellDim, dims.KDim, dtype=wpfloat),
            qg=data_alloc.constant_field(grid, 2.05496e-11, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw=data_alloc.constant_field(
                grid, -0.00234674, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            dvsi=data_alloc.constant_field(
                grid, -0.00261576, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            dvsw0=data_alloc.constant_field(
                grid, -0.00076851, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            dt=30.0,
            exchange_rate=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )

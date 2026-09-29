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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import ice_to_snow
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestIceToSnowDefault(StencilTest):
    PROGRAM = ice_to_snow
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        qi: np.ndarray,
        ns: np.ndarray,
        lam: np.ndarray,
        sticking_eff: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(conversion_rate=np.full(qi.shape, 0.0))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            qi=data_alloc.constant_field(grid, 7.95122e-25, dims.CellDim, dims.KDim, dtype=wpfloat),
            ns=data_alloc.constant_field(grid, 2.23336e07, dims.CellDim, dims.KDim, dtype=wpfloat),
            lam=data_alloc.constant_field(grid, 61911.1, dims.CellDim, dims.KDim, dtype=wpfloat),
            sticking_eff=data_alloc.constant_field(
                grid, 0.241568, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            conversion_rate=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )

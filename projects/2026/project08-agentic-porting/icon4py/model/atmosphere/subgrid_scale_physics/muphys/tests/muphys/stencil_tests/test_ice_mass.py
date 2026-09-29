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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import ice_mass
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestIceNumber(StencilTest):
    PROGRAM = ice_mass
    OUTPUTS = ("mass",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        qi: np.ndarray,
        ni: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(mass=np.full(qi.shape, 1.0e-12))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            qi=data_alloc.constant_field(grid, 2.02422e-23, dims.CellDim, dims.KDim, dtype=wpfloat),
            ni=data_alloc.constant_field(grid, 5.05089, dims.CellDim, dims.KDim, dtype=wpfloat),
            mass=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )

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
    ice_deposition_nucleation,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestIceDepositionNucleation(StencilTest):
    PROGRAM = ice_deposition_nucleation
    OUTPUTS = ("vapor_deposition_rate",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        t: np.ndarray,
        qc: np.ndarray,
        qi: np.ndarray,
        ni: np.ndarray,
        dvsi: np.ndarray,
        dt: wpfloat,
        **kwargs,
    ) -> dict:
        return dict(vapor_deposition_rate=np.full(t.shape, 1.6836299999999999e-13))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t=data_alloc.constant_field(grid, 160.9, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=data_alloc.constant_field(grid, 1.0e-2, dims.CellDim, dims.KDim, dtype=wpfloat),
            qi=data_alloc.constant_field(grid, 2.02422e-23, dims.CellDim, dims.KDim, dtype=wpfloat),
            ni=data_alloc.constant_field(grid, 5.05089, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsi=data_alloc.constant_field(grid, 0.0001, dims.CellDim, dims.KDim, dtype=wpfloat),
            dt=30.0,
            vapor_deposition_rate=data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
        )

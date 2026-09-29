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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import cloud_to_snow
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestCloudToSnow(StencilTest):
    PROGRAM = cloud_to_snow
    OUTPUTS = ("riming_snow_rate",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        t: np.ndarray,
        qc: np.ndarray,
        qs: np.ndarray,
        ns: np.ndarray,
        lam: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(riming_snow_rate=np.full(t.shape, 9.5431874564438999e-10))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t=data_alloc.constant_field(grid, 256.571, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=data_alloc.constant_field(grid, 3.31476e-05, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs=data_alloc.constant_field(grid, 7.47365e-06, dims.CellDim, dims.KDim, dtype=wpfloat),
            ns=data_alloc.constant_field(grid, 3.37707e07, dims.CellDim, dims.KDim, dtype=wpfloat),
            lam=data_alloc.constant_field(grid, 8989.78, dims.CellDim, dims.KDim, dtype=wpfloat),
            riming_snow_rate=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )

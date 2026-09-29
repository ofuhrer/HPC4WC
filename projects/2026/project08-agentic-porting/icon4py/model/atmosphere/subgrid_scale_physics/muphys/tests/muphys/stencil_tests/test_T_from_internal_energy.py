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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import T_from_internal_energy
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestTFromInternalEnergy(StencilTest):
    PROGRAM = T_from_internal_energy
    OUTPUTS = ("temperature",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        u: np.ndarray,
        qv: np.ndarray,
        qliq: np.ndarray,
        qice: np.ndarray,
        rho: np.ndarray,
        dz: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(temperature=np.full(u.shape, 255.75599999999997))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            u=data_alloc.constant_field(
                grid, 38265357.270336017, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            qv=data_alloc.constant_field(grid, 0.00122576, dims.CellDim, dims.KDim, dtype=wpfloat),
            qliq=data_alloc.constant_field(
                grid, 1.63837e-20, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            qice=data_alloc.constant_field(
                grid, 1.09462e-08, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            rho=data_alloc.constant_field(grid, 0.83444, dims.CellDim, dims.KDim, dtype=wpfloat),
            dz=data_alloc.constant_field(grid, 249.569, dims.CellDim, dims.KDim, dtype=wpfloat),
            temperature=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import constants as phy_const, dimension as dims, type_alias as ta
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_surface_pressure import (
    diagnose_surface_pressure,
)
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestDiagnoseSurfacePressure(stencil_tests.StencilTest):
    PROGRAM = diagnose_surface_pressure
    OUTPUTS = ("surface_pressure",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        exner: np.ndarray,
        virtual_temperature: np.ndarray,
        ddqz_z_full: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        shape = virtual_temperature.shape
        surface_pressure = np.zeros((shape[0], shape[1] + 1), dtype=ta.wpfloat)
        surface_pressure[:, -1] = phy_const.P0REF * np.exp(
            phy_const.CPD_O_RD * np.log(exner[:, -3])
            + phy_const.GRAV_O_RD
            * (
                ddqz_z_full[:, -1] / virtual_temperature[:, -1]
                + ddqz_z_full[:, -2] / virtual_temperature[:, -2]
                + 0.5 * ddqz_z_full[:, -3] / virtual_temperature[:, -3]
            )
        )
        return dict(
            surface_pressure=surface_pressure,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        low = 1.0e-2
        exner = data_alloc.random_field(grid, dims.CellDim, dims.KDim, low=1.0e-6, dtype=ta.wpfloat)
        virtual_temperature = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=low, dtype=ta.wpfloat
        )
        ddqz_z_full = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=low, dtype=ta.wpfloat
        )
        surface_pressure = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, extend={dims.KDim: 1}
        )

        return dict(
            exner=exner,
            virtual_temperature=virtual_temperature,
            ddqz_z_full=ddqz_z_full,
            surface_pressure=surface_pressure,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=gtx.int32(grid.num_levels),
            vertical_end=gtx.int32(grid.num_levels + 1),
        )

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
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_pressure import (
    diagnose_pressure,
)
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestDiagnosePressure(stencil_tests.StencilTest):
    PROGRAM = diagnose_pressure
    OUTPUTS = ("pressure", "pressure_ifc")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        surface_pressure: np.ndarray,
        virtual_temperature: np.ndarray,
        ddqz_z_full: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        pressure_ifc = np.zeros_like(virtual_temperature)
        pressure = np.zeros_like(virtual_temperature)
        ground_level = virtual_temperature.shape[1] - 1
        pressure_ifc[:, ground_level] = surface_pressure * np.exp(
            -phy_const.GRAV_O_RD
            * ddqz_z_full[:, ground_level]
            / virtual_temperature[:, ground_level]
        )
        pressure[:, ground_level] = np.sqrt(pressure_ifc[:, ground_level] * surface_pressure)
        for k in range(ground_level - 1, -1, -1):
            pressure_ifc[:, k] = pressure_ifc[:, k + 1] * np.exp(
                -phy_const.GRAV_O_RD * ddqz_z_full[:, k] / virtual_temperature[:, k]
            )
            pressure[:, k] = np.sqrt(pressure_ifc[:, k] * pressure_ifc[:, k + 1])

        return dict(
            pressure=pressure,
            pressure_ifc=pressure_ifc,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        ddqz_z_full = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=1.0, dtype=ta.wpfloat
        )
        virtual_temperature = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=1.0e-2, dtype=ta.wpfloat
        )
        surface_pressure = data_alloc.random_field(grid, dims.CellDim, low=1.0, dtype=ta.wpfloat)
        pressure = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        pressure_ifc = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)

        return dict(
            ddqz_z_full=ddqz_z_full,
            virtual_temperature=virtual_temperature,
            surface_pressure=surface_pressure,
            pressure=pressure,
            pressure_ifc=pressure_ifc,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )

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
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_temperature import (
    diagnose_virtual_temperature_and_temperature,
)
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestDiagnoseTemperature(stencil_tests.StencilTest):
    PROGRAM = diagnose_virtual_temperature_and_temperature
    OUTPUTS = ("virtual_temperature", "temperature")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        qv: np.ndarray,
        qc: np.ndarray,
        qi: np.ndarray,
        qr: np.ndarray,
        qs: np.ndarray,
        qg: np.ndarray,
        theta_v: np.ndarray,
        exner: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        qsum = qc + qi + qr + qs + qg
        virtual_temperature = theta_v * exner
        temperature = virtual_temperature / (1.0 + phy_const.RV_O_RD_MINUS_1 * qv - qsum)
        return dict(
            virtual_temperature=virtual_temperature,
            temperature=temperature,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        theta_v = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=1.0e-4, high=1.0, dtype=ta.wpfloat
        )
        exner = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=1.0e-4, high=1.0, dtype=ta.wpfloat
        )
        qv = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        qc = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        qi = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        qr = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        qs = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        qg = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        virtual_temperature = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        temperature = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)

        return dict(
            qv=qv,
            qc=qc,
            qi=qi,
            qr=qr,
            qs=qs,
            qg=qg,
            theta_v=theta_v,
            exner=exner,
            virtual_temperature=virtual_temperature,
            temperature=temperature,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )

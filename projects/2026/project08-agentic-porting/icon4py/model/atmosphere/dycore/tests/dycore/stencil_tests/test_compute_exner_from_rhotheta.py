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

from icon4py.model.atmosphere.dycore.stencils.compute_exner_from_rhotheta import (
    _compute_exner_from_rhotheta,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


class TestComputeExnerFromRhotheta(StencilTest):
    PROGRAM = _compute_exner_from_rhotheta
    OUTPUTS = ("out",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        rho: np.ndarray,
        exner: np.ndarray,
        rd_o_cvd: float,
        rd_o_p0ref: float,
        **kwargs: Any,
    ) -> dict:
        theta_v = np.copy(exner)
        exner = np.exp(rd_o_cvd * np.log(rd_o_p0ref * rho * theta_v))
        return dict(out=(theta_v, exner))

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        rd_o_cvd = wpfloat("10.0")
        rd_o_p0ref = wpfloat("20.0")
        rho = random_field(grid, dims.CellDim, dims.KDim, low=1, high=2, dtype=wpfloat)
        exner = random_field(grid, dims.CellDim, dims.KDim, low=1, high=2, dtype=wpfloat)
        theta_v = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            rho=rho,
            exner=exner,
            rd_o_cvd=rd_o_cvd,
            rd_o_p0ref=rd_o_p0ref,
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KDim: (0, gtx.int32(grid.num_levels)),
            },
            out=(theta_v, exner),
        )

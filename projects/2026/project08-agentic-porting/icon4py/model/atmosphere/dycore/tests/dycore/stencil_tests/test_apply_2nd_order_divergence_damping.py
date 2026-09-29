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

import icon4py.model.common.type_alias as ta
from icon4py.model.atmosphere.dycore.stencils.apply_2nd_order_divergence_damping import (
    apply_2nd_order_divergence_damping,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.stencil_tests import StencilTest


def apply_2nd_order_divergence_damping_numpy(
    z_graddiv_vn: np.ndarray, vn: np.ndarray, scal_divdamp_o2: wpfloat
) -> np.ndarray:
    vn = vn + (scal_divdamp_o2 * z_graddiv_vn)
    return vn


class TestApply2ndOrderDivergenceDamping(StencilTest):
    PROGRAM = apply_2nd_order_divergence_damping
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        z_graddiv_vn: np.ndarray,
        vn: np.ndarray,
        scal_divdamp_o2: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        vn = apply_2nd_order_divergence_damping_numpy(z_graddiv_vn, vn, scal_divdamp_o2)
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        z_graddiv_vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        scal_divdamp_o2 = wpfloat("5.0")

        return dict(
            z_graddiv_vn=z_graddiv_vn,
            vn=vn,
            scal_divdamp_o2=scal_divdamp_o2,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

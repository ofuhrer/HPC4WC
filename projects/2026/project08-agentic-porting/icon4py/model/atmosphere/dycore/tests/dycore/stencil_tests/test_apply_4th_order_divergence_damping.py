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

from icon4py.model.atmosphere.dycore.stencils.apply_4th_order_divergence_damping import (
    apply_4th_order_divergence_damping,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest

from . import test_dycore_utils


def apply_4th_order_divergence_damping_numpy(
    scal_divdamp: np.ndarray,
    z_graddiv2_vn: np.ndarray,
    vn: np.ndarray,
) -> np.ndarray:
    scal_divdamp = np.expand_dims(scal_divdamp, axis=0)
    vn = vn + (scal_divdamp * z_graddiv2_vn)
    return vn


class TestApply4thOrderDivergenceDamping(StencilTest):
    PROGRAM = apply_4th_order_divergence_damping
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        interpolated_fourth_order_divdamp_factor: np.ndarray,
        z_graddiv2_vn: np.ndarray,
        vn: np.ndarray,
        divdamp_order: gtx.int32,
        mean_cell_area: float,
        second_order_divdamp_factor: float,
        **kwargs: Any,
    ) -> dict:
        scal_divdamp = test_dycore_utils.fourth_order_divdamp_scaling_coeff_numpy(
            interpolated_fourth_order_divdamp_factor,
            divdamp_order,
            second_order_divdamp_factor,
            mean_cell_area,
        )
        vn = apply_4th_order_divergence_damping_numpy(scal_divdamp, z_graddiv2_vn, vn)
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        interpolated_fourth_order_divdamp_factor = data_alloc.random_field(grid, dims.KDim)
        z_graddiv2_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)

        divdamp_order = 24
        mean_cell_area = 1000.0
        second_order_divdamp_factor = 3.0

        return dict(
            interpolated_fourth_order_divdamp_factor=interpolated_fourth_order_divdamp_factor,
            z_graddiv2_vn=z_graddiv2_vn,
            vn=vn,
            divdamp_order=divdamp_order,
            mean_cell_area=mean_cell_area,
            second_order_divdamp_factor=second_order_divdamp_factor,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

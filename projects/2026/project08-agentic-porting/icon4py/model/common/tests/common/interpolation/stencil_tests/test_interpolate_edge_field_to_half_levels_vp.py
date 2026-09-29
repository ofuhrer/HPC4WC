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

import icon4py.model.testing.stencil_tests as test_helpers
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.interpolation.stencils.interpolate_edge_field_to_half_levels_vp import (
    interpolate_edge_field_to_half_levels_vp,
)
from icon4py.model.common.type_alias import vpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field


def interpolate_edge_field_to_half_levels_vp_numpy(
    wgtfac_e: np.ndarray, interpolant: np.ndarray
) -> np.ndarray:
    interpolant_offset_1 = np.roll(interpolant, shift=1, axis=1)
    interpolation_to_half_levels_vp = (
        wgtfac_e * interpolant + (1.0 - wgtfac_e) * interpolant_offset_1
    )
    interpolation_to_half_levels_vp[:, 0] = 0

    return interpolation_to_half_levels_vp


class TestInterpolateToHalfLevelsVp(test_helpers.StencilTest):
    PROGRAM = interpolate_edge_field_to_half_levels_vp
    OUTPUTS = ("interpolation_to_half_levels_vp",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        wgtfac_e: np.ndarray,
        interpolant: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        interpolation_to_half_levels_vp = interpolate_edge_field_to_half_levels_vp_numpy(
            wgtfac_e=wgtfac_e, interpolant=interpolant
        )
        return dict(interpolation_to_half_levels_vp=interpolation_to_half_levels_vp)

    @pytest.fixture
    def input_data(self, grid: base_grid.Grid) -> dict:
        interpolant = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        wgtfac_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        interpolation_to_half_levels_vp = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            wgtfac_e=wgtfac_e,
            interpolant=interpolant,
            interpolation_to_half_levels_vp=interpolation_to_half_levels_vp,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )

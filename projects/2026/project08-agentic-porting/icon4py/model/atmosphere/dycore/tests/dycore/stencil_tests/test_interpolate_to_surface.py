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

from icon4py.model.atmosphere.dycore.stencils.interpolate_to_surface import interpolate_to_surface
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def interpolate_to_surface_numpy(
    interpolant: np.ndarray, wgtfacq_c: np.ndarray, interpolation_to_surface: np.ndarray
) -> np.ndarray:
    interpolation_to_surface = np.copy(interpolation_to_surface)
    interpolation_to_surface[:, 3:] = (
        np.roll(wgtfacq_c, shift=1, axis=1) * np.roll(interpolant, shift=1, axis=1)
        + np.roll(wgtfacq_c, shift=2, axis=1) * np.roll(interpolant, shift=2, axis=1)
        + np.roll(wgtfacq_c, shift=3, axis=1) * np.roll(interpolant, shift=3, axis=1)
    )[:, 3:]
    return interpolation_to_surface


class TestInterpolateToSurface(StencilTest):
    PROGRAM = interpolate_to_surface
    OUTPUTS = ("interpolation_to_surface",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        interpolant: np.ndarray,
        wgtfacq_c: np.ndarray,
        interpolation_to_surface: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        interpolation_to_surface = interpolate_to_surface_numpy(
            wgtfacq_c=wgtfacq_c,
            interpolant=interpolant,
            interpolation_to_surface=interpolation_to_surface,
        )
        return dict(interpolation_to_surface=interpolation_to_surface)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        interpolant = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        wgtfacq_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        interpolation_to_surface = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            interpolant=interpolant,
            wgtfacq_c=wgtfacq_c,
            interpolation_to_surface=interpolation_to_surface,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=3,
            vertical_end=gtx.int32(grid.num_levels),
        )

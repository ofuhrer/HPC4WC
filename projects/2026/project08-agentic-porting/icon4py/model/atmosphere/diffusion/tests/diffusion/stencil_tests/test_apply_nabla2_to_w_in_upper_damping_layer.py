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

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w_in_upper_damping_layer import (
    apply_nabla2_to_w_in_upper_damping_layer,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.stencil_tests import StencilTest


def apply_nabla2_to_w_in_upper_damping_layer_numpy(
    w: np.ndarray,
    diff_multfac_n2w: np.ndarray,
    cell_area: np.ndarray,
    z_nabla2_c: np.ndarray,
) -> np.ndarray:
    cell_area = np.expand_dims(cell_area, axis=-1)
    w = w + diff_multfac_n2w * cell_area * z_nabla2_c
    return w


class TestApplyNabla2ToWInUpperDampingLayer(StencilTest):
    PROGRAM = apply_nabla2_to_w_in_upper_damping_layer
    OUTPUTS = ("w",)

    @pytest.fixture
    def input_data(self, grid: base.Grid):
        w = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        diff_multfac_n2w = random_field(grid, dims.KDim, dtype=wpfloat)
        cell_area = random_field(grid, dims.CellDim, dtype=wpfloat)
        z_nabla2_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            w=w,
            diff_multfac_n2w=diff_multfac_n2w,
            cell_area=cell_area,
            z_nabla2_c=z_nabla2_c,
            horizontal_start=0,
            horizontal_end=int(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        w: np.ndarray,
        diff_multfac_n2w: np.ndarray,
        cell_area: np.ndarray,
        z_nabla2_c: np.ndarray,
        **kwargs,
    ) -> dict:
        w = apply_nabla2_to_w_in_upper_damping_layer_numpy(
            w, diff_multfac_n2w, cell_area, z_nabla2_c
        )
        return dict(w=w)

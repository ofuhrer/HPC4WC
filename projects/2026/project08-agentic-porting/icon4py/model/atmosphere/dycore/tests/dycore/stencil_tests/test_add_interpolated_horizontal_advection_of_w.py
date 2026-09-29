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

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.dycore.stencils.add_interpolated_horizontal_advection_of_w import (
    add_interpolated_horizontal_advection_of_w,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing import stencil_tests


def add_interpolated_horizontal_advection_of_w_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    e_bln_c_s: np.ndarray,
    z_v_grad_w: np.ndarray,
    ddt_w_adv: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    c2e = connectivities[dims.C2EDim]

    ddt_w_adv = ddt_w_adv + np.sum(
        z_v_grad_w[c2e] * e_bln_c_s,
        axis=1,
    )
    return ddt_w_adv


class TestAddInterpolatedHorizontalAdvectionOfW(stencil_tests.StencilTest):
    PROGRAM = add_interpolated_horizontal_advection_of_w
    OUTPUTS = ("ddt_w_adv",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        e_bln_c_s: np.ndarray,
        z_v_grad_w: np.ndarray,
        ddt_w_adv: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        ddt_w_adv = add_interpolated_horizontal_advection_of_w_numpy(
            connectivities, e_bln_c_s, z_v_grad_w, ddt_w_adv
        )
        return dict(ddt_w_adv=ddt_w_adv)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        z_v_grad_w = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        e_bln_c_s = data_alloc.random_field(grid, dims.CellDim, dims.C2EDim, dtype=ta.wpfloat)
        ddt_w_adv = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)

        return dict(
            e_bln_c_s=e_bln_c_s,
            z_v_grad_w=z_v_grad_w,
            ddt_w_adv=ddt_w_adv,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

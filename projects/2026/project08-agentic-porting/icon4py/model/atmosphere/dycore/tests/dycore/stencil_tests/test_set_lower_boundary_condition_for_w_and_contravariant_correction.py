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

from icon4py.model.atmosphere.dycore.stencils.set_lower_boundary_condition_for_w_and_contravariant_correction import (
    set_lower_boundary_condition_for_w_and_contravariant_correction,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def set_lower_boundary_condition_for_w_and_contravariant_correction_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    w_concorr_c: np.ndarray,
    z_contr_w_fl_l: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    w_nnew = w_concorr_c
    z_contr_w_fl_l = np.zeros_like(z_contr_w_fl_l)
    return (w_nnew, z_contr_w_fl_l)


class TestInitLowerBoundaryConditionForWAndContravariantCorrection(StencilTest):
    PROGRAM = set_lower_boundary_condition_for_w_and_contravariant_correction
    OUTPUTS = ("w_nnew", "z_contr_w_fl_l")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        w_concorr_c: np.ndarray,
        z_contr_w_fl_l: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        (
            w_nnew,
            z_contr_w_fl_l,
        ) = set_lower_boundary_condition_for_w_and_contravariant_correction_numpy(
            connectivities, w_concorr_c, z_contr_w_fl_l
        )
        return dict(w_nnew=w_nnew, z_contr_w_fl_l=z_contr_w_fl_l)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        w_concorr_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_contr_w_fl_l = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        w_nnew = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            w_nnew=w_nnew,
            z_contr_w_fl_l=z_contr_w_fl_l,
            w_concorr_c=w_concorr_c,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

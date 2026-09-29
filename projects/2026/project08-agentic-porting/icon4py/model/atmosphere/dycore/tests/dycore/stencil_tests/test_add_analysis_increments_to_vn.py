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
from icon4py.model.atmosphere.dycore.stencils.add_analysis_increments_to_vn import (
    add_analysis_increments_to_vn,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.stencil_tests import StencilTest


def add_analysis_increments_to_vn_numpy(
    vn_incr: np.ndarray, vn: np.ndarray, iau_wgt_dyn: ta.wpfloat
) -> np.ndarray:
    vn = vn + (iau_wgt_dyn * vn_incr)
    return vn


class TestAddAnalysisIncrementsToVn(StencilTest):
    PROGRAM = add_analysis_increments_to_vn
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        vn_incr: np.ndarray,
        vn: np.ndarray,
        iau_wgt_dyn: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        vn = add_analysis_increments_to_vn_numpy(vn_incr, vn, iau_wgt_dyn)
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        vn_incr = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        iau_wgt_dyn = wpfloat("5.0")

        return dict(
            vn_incr=vn_incr,
            vn=vn,
            iau_wgt_dyn=iau_wgt_dyn,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

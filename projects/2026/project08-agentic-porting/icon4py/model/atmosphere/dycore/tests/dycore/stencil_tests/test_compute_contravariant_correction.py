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

from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction import (
    compute_contravariant_correction,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def compute_contravariant_correction_numpy(
    vn: np.ndarray, ddxn_z_full: np.ndarray, ddxt_z_full: np.ndarray, vt: np.ndarray
) -> np.ndarray:
    z_w_concorr_me = vn * ddxn_z_full + vt * ddxt_z_full
    return z_w_concorr_me


class TestComputeContravariantCorrection(StencilTest):
    PROGRAM = compute_contravariant_correction
    OUTPUTS = ("z_w_concorr_me",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        vn: np.ndarray,
        ddxn_z_full: np.ndarray,
        ddxt_z_full: np.ndarray,
        vt: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        z_w_concorr_me = compute_contravariant_correction_numpy(vn, ddxn_z_full, ddxt_z_full, vt)
        return dict(z_w_concorr_me=z_w_concorr_me)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        ddxn_z_full = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        ddxt_z_full = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat, low=0.1)
        vt = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        z_w_concorr_me = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            vn=vn,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            vt=vt,
            z_w_concorr_me=z_w_concorr_me,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

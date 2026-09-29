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

from icon4py.model.atmosphere.dycore.stencils.compute_tangential_wind import compute_tangential_wind
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def compute_tangential_wind_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray], vn: np.ndarray, rbf_vec_coeff_e: np.ndarray
) -> np.ndarray:
    rbf_vec_coeff_e = np.expand_dims(rbf_vec_coeff_e, axis=-1)
    e2c2e = connectivities[dims.E2C2EDim]
    vt = np.sum(np.where((e2c2e != -1)[:, :, np.newaxis], vn[e2c2e] * rbf_vec_coeff_e, 0), axis=1)
    return vt


@pytest.mark.embedded_remap_error
class TestComputeTangentialWind(StencilTest):
    PROGRAM = compute_tangential_wind
    OUTPUTS = ("vt",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        vn: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        vt = compute_tangential_wind_numpy(connectivities, vn, rbf_vec_coeff_e)
        return dict(vt=vt)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        rbf_vec_coeff_e = random_field(grid, dims.EdgeDim, dims.E2C2EDim, dtype=wpfloat)
        vt = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            vn=vn,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            vt=vt,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

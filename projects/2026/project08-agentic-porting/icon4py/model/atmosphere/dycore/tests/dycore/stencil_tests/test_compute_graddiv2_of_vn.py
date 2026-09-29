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

from icon4py.model.atmosphere.dycore.stencils.compute_graddiv2_of_vn import compute_graddiv2_of_vn
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def compute_graddiv2_of_vn_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    geofac_grdiv: np.ndarray,
    z_graddiv_vn: np.ndarray,
) -> np.ndarray:
    e2c2eO = connectivities[dims.E2C2EODim]
    geofac_grdiv = np.expand_dims(geofac_grdiv, axis=-1)
    z_graddiv2_vn = np.sum(
        np.where((e2c2eO != -1)[:, :, np.newaxis], z_graddiv_vn[e2c2eO] * geofac_grdiv, 0),
        axis=1,
    )
    return z_graddiv2_vn


@pytest.mark.embedded_remap_error
class TestComputeGraddiv2OfVn(StencilTest):
    PROGRAM = compute_graddiv2_of_vn
    OUTPUTS = ("z_graddiv2_vn",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        geofac_grdiv: np.ndarray,
        z_graddiv_vn: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        z_graddiv2_vn = compute_graddiv2_of_vn_numpy(connectivities, geofac_grdiv, z_graddiv_vn)
        return dict(z_graddiv2_vn=z_graddiv2_vn)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        z_graddiv_vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        geofac_grdiv = random_field(grid, dims.EdgeDim, dims.E2C2EODim, dtype=wpfloat)
        z_graddiv2_vn = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            geofac_grdiv=geofac_grdiv,
            z_graddiv_vn=z_graddiv_vn,
            z_graddiv2_vn=z_graddiv2_vn,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

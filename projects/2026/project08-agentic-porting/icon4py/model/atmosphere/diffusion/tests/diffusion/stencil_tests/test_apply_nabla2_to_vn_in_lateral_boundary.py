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

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_vn_in_lateral_boundary import (
    apply_nabla2_to_vn_in_lateral_boundary,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.stencil_tests import StencilTest


def apply_nabla2_to_vn_in_lateral_boundary_numpy(
    z_nabla2_e: np.array, area_edge: np.array, vn: np.array, fac_bdydiff_v
) -> np.array:
    area_edge = np.expand_dims(area_edge, axis=-1)
    vn = vn + (z_nabla2_e * area_edge * fac_bdydiff_v)
    return vn


class TestApplyNabla2ToVnInLateralBoundary(StencilTest):
    PROGRAM = apply_nabla2_to_vn_in_lateral_boundary
    OUTPUTS = ("vn",)

    @pytest.fixture
    def input_data(self, grid: base.Grid):
        fac_bdydiff_v = wpfloat("5.0")
        z_nabla2_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        area_edge = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        return dict(
            fac_bdydiff_v=fac_bdydiff_v,
            z_nabla2_e=z_nabla2_e,
            area_edge=area_edge,
            vn=vn,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        z_nabla2_e: np.ndarray,
        area_edge: np.ndarray,
        vn: np.ndarray,
        fac_bdydiff_v: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        vn = apply_nabla2_to_vn_in_lateral_boundary_numpy(z_nabla2_e, area_edge, vn, fac_bdydiff_v)
        return dict(vn=vn)

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

from icon4py.model.atmosphere.dycore.stencils.interpolate_vt_to_interface_edges import (
    interpolate_vt_to_interface_edges,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def interpolate_vt_to_interface_edges_numpy(
    wgtfac_e: np.ndarray, vt: np.ndarray, **kwargs: Any
) -> np.ndarray:
    vt_k_minus_1 = np.roll(vt, shift=1, axis=1)
    z_vt_ie = wgtfac_e * vt + (1.0 - wgtfac_e) * vt_k_minus_1
    z_vt_ie[:, 0] = vt[:, 0]
    return z_vt_ie


class TestInterpolateVtToInterfaceEdges(StencilTest):
    PROGRAM = interpolate_vt_to_interface_edges
    OUTPUTS = ("z_vt_ie",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        wgtfac_e: np.ndarray,
        vt: np.ndarray,
        z_vt_ie: np.ndarray,
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
        vertical_start: gtx.int32,
        vertical_end: gtx.int32,
        **kwargs: Any,
    ) -> dict:
        subset = (slice(horizontal_start, horizontal_end), slice(vertical_start, vertical_end))
        z_vt_ie = z_vt_ie.copy()
        z_vt_ie[subset] = interpolate_vt_to_interface_edges_numpy(wgtfac_e, vt)[subset]
        return dict(z_vt_ie=z_vt_ie)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        wgtfac_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        vt = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)

        z_vt_ie = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)

        return dict(
            wgtfac_e=wgtfac_e,
            vt=vt,
            z_vt_ie=z_vt_ie,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )

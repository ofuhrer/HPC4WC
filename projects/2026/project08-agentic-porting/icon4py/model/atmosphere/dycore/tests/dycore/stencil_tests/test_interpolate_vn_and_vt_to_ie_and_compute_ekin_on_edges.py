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

from icon4py.model.atmosphere.dycore.stencils.interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges import (
    interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


class TestInterpolateVnAndVtToIeAndComputeEkinOnEdges(StencilTest):
    PROGRAM = interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges
    OUTPUTS = ("vn_ie", "z_vt_ie", "z_kin_hor_e")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        wgtfac_e: np.ndarray,
        vn: np.ndarray,
        vt: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        vn_offset_1 = np.roll(vn, shift=1, axis=1)
        vt_offset_1 = np.roll(vt, shift=1, axis=1)

        vn_ie = wgtfac_e * vn + (1.0 - wgtfac_e) * vn_offset_1
        vn_ie[:, 0] = 0
        z_vt_ie = wgtfac_e * vt + (1.0 - wgtfac_e) * vt_offset_1
        z_vt_ie[:, 0] = 0
        z_kin_hor_e = 0.5 * (vn**2 + vt**2)
        z_kin_hor_e[:, 0] = 0
        return dict(vn_ie=vn_ie, z_vt_ie=z_vt_ie, z_kin_hor_e=z_kin_hor_e)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        wgtfac_e = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        vt = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vn_ie = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        z_vt_ie = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        z_kin_hor_e = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            wgtfac_e=wgtfac_e,
            vn=vn,
            vt=vt,
            vn_ie=vn_ie,
            z_vt_ie=z_vt_ie,
            z_kin_hor_e=z_kin_hor_e,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )

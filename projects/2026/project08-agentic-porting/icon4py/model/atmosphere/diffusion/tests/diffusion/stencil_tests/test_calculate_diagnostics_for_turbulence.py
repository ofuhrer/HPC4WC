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

from icon4py.model.atmosphere.diffusion.stencils.calculate_diagnostics_for_turbulence import (
    calculate_diagnostics_for_turbulence,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def calculate_diagnostics_for_turbulence_numpy(
    wgtfac_c: np.ndarray, div: np.ndarray, kh_c: np.ndarray, div_ic, hdef_ic
) -> tuple[np.ndarray, np.ndarray]:
    kc_offset_1 = np.roll(kh_c, shift=1, axis=1)
    div_offset_1 = np.roll(div, shift=1, axis=1)
    div_ic[:, 1:] = (wgtfac_c * div + (1.0 - wgtfac_c) * div_offset_1)[:, 1:]
    hdef_ic[:, 1:] = ((wgtfac_c * kh_c + (1.0 - wgtfac_c) * kc_offset_1) ** 2)[:, 1:]
    return div_ic, hdef_ic


class TestCalculateDiagnosticsForTurbulence(StencilTest):
    PROGRAM = calculate_diagnostics_for_turbulence
    OUTPUTS = ("div_ic", "hdef_ic")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        wgtfac_c: np.ndarray,
        div: np.ndarray,
        kh_c: np.ndarray,
        div_ic: np.ndarray,
        hdef_ic: np.ndarray,
        **kwargs: object,
    ) -> dict:
        div_ic, hdef_ic = calculate_diagnostics_for_turbulence_numpy(
            wgtfac_c, div, kh_c, div_ic, hdef_ic
        )
        return dict(div_ic=div_ic, hdef_ic=hdef_ic)

    @pytest.fixture
    def input_data(self, grid):
        wgtfac_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        div = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        kh_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        div_ic = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        hdef_ic = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        return dict(wgtfac_c=wgtfac_c, div=div, kh_c=kh_c, div_ic=div_ic, hdef_ic=hdef_ic)

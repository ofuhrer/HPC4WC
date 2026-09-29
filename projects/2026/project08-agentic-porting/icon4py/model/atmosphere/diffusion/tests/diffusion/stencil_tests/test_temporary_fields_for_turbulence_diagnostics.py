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

from icon4py.model.atmosphere.diffusion.stencils.temporary_fields_for_turbulence_diagnostics import (
    temporary_fields_for_turbulence_diagnostics,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestTemporaryFieldsForTurbulenceDiagnostics(stencil_tests.StencilTest):
    PROGRAM = temporary_fields_for_turbulence_diagnostics
    OUTPUTS = ("div", "kh_c")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        kh_smag_ec: np.ndarray,
        vn: np.ndarray,
        e_bln_c_s: np.ndarray,
        geofac_div: np.ndarray,
        diff_multfac_smag: np.ndarray,
        **kwargs,
    ) -> dict:
        c2e = connectivities[dims.C2EDim]

        geofac_div = np.expand_dims(geofac_div, axis=-1)
        e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
        diff_multfac_smag = np.expand_dims(diff_multfac_smag, axis=0)

        vn_geofac = vn[c2e] * geofac_div
        div = np.sum(vn_geofac, axis=1)
        mul = kh_smag_ec[c2e] * e_bln_c_s
        summed = np.sum(mul, axis=1)
        kh_c = summed / diff_multfac_smag

        return dict(div=div, kh_c=kh_c)

    @pytest.fixture
    def input_data(self, grid: base.Grid):
        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        geofac_div = data_alloc.random_field(grid, dims.CellDim, dims.C2EDim, dtype=ta.wpfloat)

        kh_smag_ec = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        e_bln_c_s = data_alloc.random_field(grid, dims.CellDim, dims.C2EDim, dtype=ta.wpfloat)

        diff_multfac_smag = data_alloc.random_field(grid, dims.KDim, dtype=ta.vpfloat)

        kh_c = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        div = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)

        return dict(
            kh_smag_ec=kh_smag_ec,
            vn=vn,
            e_bln_c_s=e_bln_c_s,
            geofac_div=geofac_div,
            diff_multfac_smag=diff_multfac_smag,
            kh_c=kh_c,
            div=div,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

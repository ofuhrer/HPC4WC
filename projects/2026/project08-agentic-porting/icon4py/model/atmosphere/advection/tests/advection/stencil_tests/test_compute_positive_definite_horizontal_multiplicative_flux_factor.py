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

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.advection.stencils.compute_positive_definite_horizontal_multiplicative_flux_factor import (
    compute_positive_definite_horizontal_multiplicative_flux_factor,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing import stencil_tests


class TestComputePositiveDefiniteHorizontalMultiplicativeFluxFactor(stencil_tests.StencilTest):
    PROGRAM = compute_positive_definite_horizontal_multiplicative_flux_factor
    OUTPUTS = ("r_m",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        geofac_div: np.ndarray,
        p_cc: np.ndarray,
        p_rhodz_now: np.ndarray,
        p_mflx_tracer_h: np.ndarray,
        p_dtime,
        dbl_eps,
        **kwargs,
    ) -> dict:
        c2e = connectivities[dims.C2EDim]
        geofac_div = np.expand_dims(geofac_div, axis=-1)
        p_m_0 = np.maximum(
            0.0,
            p_mflx_tracer_h[c2e[:, 0]] * geofac_div[:, 0] * p_dtime,
        )
        p_m_1 = np.maximum(
            0.0,
            p_mflx_tracer_h[c2e[:, 1]] * geofac_div[:, 1] * p_dtime,
        )
        p_m_2 = np.maximum(
            0.0,
            p_mflx_tracer_h[c2e[:, 2]] * geofac_div[:, 2] * p_dtime,
        )

        p_m = p_m_0 + p_m_1 + p_m_2
        r_m = np.minimum(1.0, p_cc * p_rhodz_now / (p_m + dbl_eps))

        return dict(r_m=r_m)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        geofac_div = data_alloc.random_field(grid, dims.CellDim, dims.C2EDim)
        p_cc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_rhodz_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_mflx_tracer_h = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        r_m = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_dtime = np.float64(5)
        dbl_eps = np.float64(1e-9)
        return dict(
            geofac_div=geofac_div,
            p_cc=p_cc,
            p_rhodz_now=p_rhodz_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            p_dtime=p_dtime,
            dbl_eps=dbl_eps,
            r_m=r_m,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

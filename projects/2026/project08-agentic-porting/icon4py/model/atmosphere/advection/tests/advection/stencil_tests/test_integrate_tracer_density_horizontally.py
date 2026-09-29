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

from icon4py.model.atmosphere.advection.stencils.integrate_tracer_density_horizontally import (
    integrate_tracer_density_horizontally,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestIntegrateTracerDensityHorizontally(stencil_tests.StencilTest):
    PROGRAM = integrate_tracer_density_horizontally
    OUTPUTS = (
        "z_rhofluxdiv_c_out",
        "z_fluxdiv_c_dsl",
        "z_rho_new_dsl",
        "z_tracer_new_dsl",
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_mass_flx_e: np.ndarray,
        geofac_div: np.ndarray,
        z_rhofluxdiv_c: np.ndarray,
        z_tracer_mflx: np.ndarray,
        z_rho_now: np.ndarray,
        z_tracer_now: np.ndarray,
        z_dtsub: float,
        nsub: gtx.int32,
        **kwargs: Any,
    ) -> dict:
        c2e = connectivities[dims.C2EDim]
        p_mass_flx_e_c2e = p_mass_flx_e[c2e]
        geofac_div = np.expand_dims(geofac_div, axis=-1)
        z_tracer_mflx_c2e = z_tracer_mflx[c2e]

        z_rhofluxdiv_c_out = (
            np.sum(p_mass_flx_e_c2e * geofac_div, axis=1) if nsub == 1 else z_rhofluxdiv_c
        )
        z_fluxdiv_c_dsl = np.sum(z_tracer_mflx_c2e * geofac_div, axis=1)
        z_rho_new_dsl = z_rho_now - z_dtsub * z_rhofluxdiv_c_out
        z_tracer_new_dsl = (z_tracer_now * z_rho_now - z_dtsub * z_fluxdiv_c_dsl) / z_rho_new_dsl

        return dict(
            z_rhofluxdiv_c_out=z_rhofluxdiv_c_out,
            z_fluxdiv_c_dsl=z_fluxdiv_c_dsl,
            z_rho_new_dsl=z_rho_new_dsl,
            z_tracer_new_dsl=z_tracer_new_dsl,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        p_mass_flx_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        geofac_div = data_alloc.random_field(grid, dims.CellDim, dims.C2EDim)
        z_rhofluxdiv_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_tracer_mflx = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_rho_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_tracer_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_rhofluxdiv_c_out = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        z_fluxdiv_c_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        z_rho_new_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        z_tracer_new_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        z_dtsub = 0.5
        nsub = 1
        return dict(
            p_mass_flx_e=p_mass_flx_e,
            geofac_div=geofac_div,
            z_rhofluxdiv_c=z_rhofluxdiv_c,
            z_tracer_mflx=z_tracer_mflx,
            z_rho_now=z_rho_now,
            z_tracer_now=z_tracer_now,
            z_rhofluxdiv_c_out=z_rhofluxdiv_c_out,
            z_fluxdiv_c_dsl=z_fluxdiv_c_dsl,
            z_rho_new_dsl=z_rho_new_dsl,
            z_tracer_new_dsl=z_tracer_new_dsl,
            z_dtsub=z_dtsub,
            nsub=nsub,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

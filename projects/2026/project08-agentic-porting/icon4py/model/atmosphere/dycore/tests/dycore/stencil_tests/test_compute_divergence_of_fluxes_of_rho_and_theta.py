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

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.dycore.stencils.compute_divergence_of_fluxes_of_rho_and_theta import (
    compute_divergence_of_fluxes_of_rho_and_theta,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing import stencil_tests


def compute_divergence_of_fluxes_of_rho_and_theta_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    geofac_div: np.ndarray,
    mass_flux_at_edges_on_model_levels: np.ndarray,
    theta_v_flux_at_edges_on_model_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c2e = connectivities[dims.C2EDim]
    geofac_div = np.expand_dims(geofac_div, axis=-1)

    divergence_of_mass_wp = np.sum(geofac_div * mass_flux_at_edges_on_model_levels[c2e], axis=1)
    divergence_of_theta_v_wp = np.sum(
        geofac_div * theta_v_flux_at_edges_on_model_levels[c2e], axis=1
    )
    return (divergence_of_mass_wp, divergence_of_theta_v_wp)


class TestComputeDivergenceConnectivityOfFluxesOfRhoAndTheta(stencil_tests.StencilTest):
    PROGRAM = compute_divergence_of_fluxes_of_rho_and_theta
    OUTPUTS = ("z_flxdiv_mass", "z_flxdiv_theta")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        geofac_div: np.ndarray,
        mass_fl_e: np.ndarray,
        z_theta_v_fl_e: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        z_flxdiv_mass, z_flxdiv_theta = compute_divergence_of_fluxes_of_rho_and_theta_numpy(
            connectivities,
            geofac_div,
            mass_fl_e,
            z_theta_v_fl_e,
        )
        return dict(z_flxdiv_mass=z_flxdiv_mass, z_flxdiv_theta=z_flxdiv_theta)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        geofac_div = data_alloc.random_field(grid, dims.CellDim, dims.C2EDim, dtype=ta.wpfloat)
        z_theta_v_fl_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        z_flxdiv_theta = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        mass_fl_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        z_flxdiv_mass = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)

        return dict(
            geofac_div=geofac_div,
            mass_fl_e=mass_fl_e,
            z_theta_v_fl_e=z_theta_v_fl_e,
            z_flxdiv_mass=z_flxdiv_mass,
            z_flxdiv_theta=z_flxdiv_theta,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

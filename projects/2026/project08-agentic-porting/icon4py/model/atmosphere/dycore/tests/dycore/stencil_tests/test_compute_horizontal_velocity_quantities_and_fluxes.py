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
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_velocity_quantities import (
    compute_horizontal_velocity_quantities_and_fluxes,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing import stencil_tests

from .test_compute_avg_vn_and_graddiv_vn_and_vt import compute_avg_vn_and_graddiv_vn_and_vt_numpy
from .test_compute_contravariant_correction import compute_contravariant_correction_numpy
from .test_compute_diagnostics_from_normal_wind import extrapolate_to_surface_numpy
from .test_compute_horizontal_kinetic_energy import compute_horizontal_kinetic_energy_numpy
from .test_compute_mass_flux import compute_mass_flux_numpy
from .test_interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges import (
    interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_numpy,
)
from .test_interpolate_vt_to_interface_edges import interpolate_vt_to_interface_edges_numpy


def compute_vt_vn_on_half_levels_and_kinetic_energy_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    vn: np.ndarray,
    tangential_wind: np.ndarray,
    vn_on_half_levels: np.ndarray,
    tangential_wind_on_half_levels: np.ndarray,
    horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
    wgtfac_e: np.ndarray,
    wgtfacq_e: np.ndarray,
    nlevp1: int,
) -> tuple[np.ndarray, ...]:
    k = np.arange(nlevp1)[np.newaxis, :]
    k_nlev = k[:, :-1]

    vn_on_half_levels[:, :-1], horizontal_kinetic_energy_at_edges_on_model_levels = np.where(
        k_nlev >= 1,
        interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_numpy(
            wgtfac_e, vn, tangential_wind
        ),
        (vn_on_half_levels[:, :-1], horizontal_kinetic_energy_at_edges_on_model_levels),
    )

    tangential_wind_on_half_levels = np.where(
        k_nlev >= 1,
        interpolate_vt_to_interface_edges_numpy(wgtfac_e, tangential_wind),
        tangential_wind_on_half_levels,
    )

    (
        vn_on_half_levels[:, :-1],
        tangential_wind_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
    ) = np.where(
        k_nlev == 0,
        compute_horizontal_kinetic_energy_numpy(vn, tangential_wind),
        (
            vn_on_half_levels[:, :-1],
            tangential_wind_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
        ),
    )

    vn_on_half_levels[:, -1] = extrapolate_to_surface_numpy(wgtfacq_e, vn)

    return (
        vn_on_half_levels,
        tangential_wind_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.continuous_benchmarking
class TestComputeHorizontalVelocityQuantitiesAndFluxes(stencil_tests.StencilTest):
    PROGRAM = compute_horizontal_velocity_quantities_and_fluxes
    OUTPUTS = (
        "spatially_averaged_vn",
        "horizontal_gradient_of_normal_wind_divergence",
        "tangential_wind",
        "mass_flux_at_edges_on_model_levels",
        "theta_v_flux_at_edges_on_model_levels",
        "vn_on_half_levels",
        "tangential_wind_on_half_levels",
        "horizontal_kinetic_energy_at_edges_on_model_levels",
        "contravariant_correction_at_edges_on_model_levels",
    )
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
            "nflatlev",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "nflatlev",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        spatially_averaged_vn: np.ndarray,
        horizontal_gradient_of_normal_wind_divergence: np.ndarray,
        tangential_wind: np.ndarray,
        mass_flux_at_edges_on_model_levels: np.ndarray,
        theta_v_flux_at_edges_on_model_levels: np.ndarray,
        tangential_wind_on_half_levels: np.ndarray,
        vn_on_half_levels: np.ndarray,
        horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
        contravariant_correction_at_edges_on_model_levels: np.ndarray,
        vn: np.ndarray,
        e_flx_avg: np.ndarray,
        geofac_grdiv: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        rho_at_edges_on_model_levels: np.ndarray,
        theta_v_at_edges_on_model_levels: np.ndarray,
        ddqz_z_full_e: np.ndarray,
        ddxn_z_full: np.ndarray,
        ddxt_z_full: np.ndarray,
        wgtfac_e: np.ndarray,
        wgtfacq_e: np.ndarray,
        nflatlev: gtx.int32,
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
        vertical_start: gtx.int32,
        vertical_end: gtx.int32,
        **kwargs: Any,
    ) -> dict:
        k = np.arange(vertical_end)[np.newaxis, :]
        k_nlev = k[:, :-1]

        initial_spatially_averaged_vn = spatially_averaged_vn.copy()
        initial_horizontal_gradient_of_normal_wind_divergence = (
            horizontal_gradient_of_normal_wind_divergence.copy()
        )
        initial_tangential_wind = tangential_wind.copy()
        initial_mass_flux_at_edges_on_model_levels = mass_flux_at_edges_on_model_levels.copy()
        initial_theta_v_flux_at_edges_on_model_levels = theta_v_flux_at_edges_on_model_levels.copy()
        initial_vn_on_half_levels = vn_on_half_levels.copy()
        initial_tangential_wind_on_half_levels = tangential_wind_on_half_levels.copy()
        initial_horizontal_kinetic_energy_at_edges_on_model_levels = (
            horizontal_kinetic_energy_at_edges_on_model_levels.copy()
        )
        initial_contravariant_correction_at_edges_on_model_levels = (
            contravariant_correction_at_edges_on_model_levels.copy()
        )

        (
            spatially_averaged_vn,
            horizontal_gradient_of_normal_wind_divergence,
            tangential_wind,
        ) = compute_avg_vn_and_graddiv_vn_and_vt_numpy(
            connectivities=connectivities,
            e_flx_avg=e_flx_avg,
            vn=vn,
            geofac_grdiv=geofac_grdiv,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
        )

        (
            mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels,
        ) = compute_mass_flux_numpy(
            rho_at_edges_on_model_levels,
            spatially_averaged_vn,
            ddqz_z_full_e,
            theta_v_at_edges_on_model_levels,
        )

        contravariant_correction_at_edges_on_model_levels = np.where(
            k_nlev >= nflatlev,
            compute_contravariant_correction_numpy(vn, ddxn_z_full, ddxt_z_full, tangential_wind),
            contravariant_correction_at_edges_on_model_levels,
        )

        (
            vn_on_half_levels,
            tangential_wind_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
        ) = compute_vt_vn_on_half_levels_and_kinetic_energy_numpy(
            connectivities=connectivities,
            vn=vn,
            tangential_wind=tangential_wind,
            vn_on_half_levels=vn_on_half_levels,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            wgtfac_e=wgtfac_e,
            wgtfacq_e=wgtfacq_e,
            nlevp1=vertical_end,
        )

        spatially_averaged_vn[:horizontal_start, :] = initial_spatially_averaged_vn[
            :horizontal_start, :
        ]
        spatially_averaged_vn[horizontal_end:, :] = initial_spatially_averaged_vn[
            horizontal_end:, :
        ]

        horizontal_gradient_of_normal_wind_divergence[:horizontal_start, :] = (
            initial_horizontal_gradient_of_normal_wind_divergence[:horizontal_start, :]
        )
        horizontal_gradient_of_normal_wind_divergence[horizontal_end:, :] = (
            initial_horizontal_gradient_of_normal_wind_divergence[horizontal_end:, :]
        )

        tangential_wind[:horizontal_start, :] = initial_tangential_wind[:horizontal_start, :]
        tangential_wind[horizontal_end:, :] = initial_tangential_wind[horizontal_end:, :]

        mass_flux_at_edges_on_model_levels[:horizontal_start, :] = (
            initial_mass_flux_at_edges_on_model_levels[:horizontal_start, :]
        )
        mass_flux_at_edges_on_model_levels[horizontal_end:, :] = (
            initial_mass_flux_at_edges_on_model_levels[horizontal_end:, :]
        )

        theta_v_flux_at_edges_on_model_levels[:horizontal_start, :] = (
            initial_theta_v_flux_at_edges_on_model_levels[:horizontal_start, :]
        )
        theta_v_flux_at_edges_on_model_levels[horizontal_end:, :] = (
            initial_theta_v_flux_at_edges_on_model_levels[horizontal_end:, :]
        )

        vn_on_half_levels[:horizontal_start, :] = initial_vn_on_half_levels[:horizontal_start, :]
        vn_on_half_levels[horizontal_end:, :] = initial_vn_on_half_levels[horizontal_end:, :]

        tangential_wind_on_half_levels[:horizontal_start, :] = (
            initial_tangential_wind_on_half_levels[:horizontal_start, :]
        )
        tangential_wind_on_half_levels[horizontal_end:, :] = initial_tangential_wind_on_half_levels[
            horizontal_end:, :
        ]

        horizontal_kinetic_energy_at_edges_on_model_levels[:horizontal_start, :] = (
            initial_horizontal_kinetic_energy_at_edges_on_model_levels[:horizontal_start, :]
        )
        horizontal_kinetic_energy_at_edges_on_model_levels[horizontal_end:, :] = (
            initial_horizontal_kinetic_energy_at_edges_on_model_levels[horizontal_end:, :]
        )

        contravariant_correction_at_edges_on_model_levels[:horizontal_start, :] = (
            initial_contravariant_correction_at_edges_on_model_levels[:horizontal_start, :]
        )
        contravariant_correction_at_edges_on_model_levels[horizontal_end:, :] = (
            initial_contravariant_correction_at_edges_on_model_levels[horizontal_end:, :]
        )

        return dict(
            spatially_averaged_vn=spatially_averaged_vn,
            horizontal_gradient_of_normal_wind_divergence=horizontal_gradient_of_normal_wind_divergence,
            tangential_wind=tangential_wind,
            mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
            vn_on_half_levels=vn_on_half_levels,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        spatially_averaged_vn = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        horizontal_gradient_of_normal_wind_divergence = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim
        )
        tangential_wind = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        mass_flux_at_edges_on_model_levels = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        theta_v_flux_at_edges_on_model_levels = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        tangential_wind_on_half_levels = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        vn_on_half_levels = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}
        )
        horizontal_kinetic_energy_at_edges_on_model_levels = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim
        )
        contravariant_correction_at_edges_on_model_levels = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim
        )

        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        wgtfac_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        wgtfacq_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        e_flx_avg = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EODim)
        geofac_grdiv = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EODim)
        rbf_vec_coeff_e = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EDim)
        rho_at_edges_on_model_levels = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        theta_v_at_edges_on_model_levels = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddqz_z_full_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddxn_z_full = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddxt_z_full = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)

        nflatlev = 5  # value is set to reflect the MCH ch1 experiment. Changing this value will change the expected runtime

        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))

        return dict(
            spatially_averaged_vn=spatially_averaged_vn,
            horizontal_gradient_of_normal_wind_divergence=horizontal_gradient_of_normal_wind_divergence,
            tangential_wind=tangential_wind,
            mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            vn=vn,
            e_flx_avg=e_flx_avg,
            geofac_grdiv=geofac_grdiv,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            rho_at_edges_on_model_levels=rho_at_edges_on_model_levels,
            theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
            ddqz_z_full_e=ddqz_z_full_e,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            wgtfac_e=wgtfac_e,
            wgtfacq_e=wgtfacq_e,
            nflatlev=nflatlev,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=grid.num_levels + 1,
        )

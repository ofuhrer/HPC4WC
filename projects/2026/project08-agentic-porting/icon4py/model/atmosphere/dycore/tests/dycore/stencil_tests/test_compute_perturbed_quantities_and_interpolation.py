# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.dycore_states import HorizontalPressureDiscretizationType
from icon4py.model.atmosphere.dycore.stencils.compute_cell_diagnostics_for_dycore import (
    compute_perturbed_quantities_and_interpolation,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests

from .test_compute_approx_of_2nd_vertical_derivative_of_exner import (
    compute_approx_of_2nd_vertical_derivative_of_exner_numpy,
)
from .test_compute_perturbation_of_rho_and_theta import compute_perturbation_of_rho_and_theta_numpy
from .test_compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers import (
    compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers_numpy,
)
from .test_compute_virtual_potential_temperatures_and_pressure_gradient import (
    compute_virtual_potential_temperatures_and_pressure_gradient_numpy,
)
from .test_extrapolate_temporally_exner_pressure import extrapolate_temporally_exner_pressure_numpy
from .test_interpolate_cell_field_to_half_levels_vp import (
    interpolate_cell_field_to_half_levels_vp_numpy,
)
from .test_interpolate_to_surface import interpolate_to_surface_numpy
from .test_set_theta_v_prime_ic_at_lower_boundary import (
    set_theta_v_prime_ic_at_lower_boundary_numpy,
)


def compute_first_vertical_derivative_numpy(
    cell_kdim_field: np.ndarray, inv_ddqz_z_full: np.ndarray
) -> np.ndarray:
    first_vertical_derivative = (cell_kdim_field[:, :-1] - cell_kdim_field[:, 1:]) * inv_ddqz_z_full
    return first_vertical_derivative


@pytest.mark.uses_concat_where
@pytest.mark.continuous_benchmarking
class TestComputePerturbedQuantitiesAndInterpolation(stencil_tests.StencilTest):
    PROGRAM = compute_perturbed_quantities_and_interpolation
    OUTPUTS = (
        "temporal_extrapolation_of_perturbed_exner",
        "perturbed_exner_at_cells_on_model_levels",
        "exner_at_cells_on_half_levels",
        "ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels",
        "perturbed_rho_at_cells_on_model_levels",
        "perturbed_theta_v_at_cells_on_model_levels",
        "rho_at_cells_on_half_levels",
        "perturbed_theta_v_at_cells_on_half_levels",
        "theta_v_at_cells_on_half_levels",
        "nonhydro_buoy_at_cells_on_half_levels",
        "d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels",
    )
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "igradp_method",
            "nflatlev",
            "nflat_gradp",
            "start_cell_lateral_boundary",
            "start_cell_lateral_boundary_level_3",
            "start_cell_halo_level_2",
            "end_cell_halo",
            "end_cell_halo_level_2",
            "model_top",
            "surface_level",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "igradp_method",
            "nflatlev",
            "nflat_gradp",
            "model_top",
            "surface_level",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        current_rho: np.ndarray,
        reference_rho_at_cells_on_model_levels: np.ndarray,
        current_theta_v: np.ndarray,
        reference_theta_at_cells_on_model_levels: np.ndarray,
        perturbed_rho_at_cells_on_model_levels: np.ndarray,
        perturbed_theta_v_at_cells_on_model_levels: np.ndarray,
        perturbed_theta_v_at_cells_on_half_levels: np.ndarray,
        reference_theta_at_cells_on_half_levels: np.ndarray,
        wgtfacq_c: np.ndarray,
        wgtfac_c: np.ndarray,
        exner_w_explicit_weight_parameter: np.ndarray,
        perturbed_exner_at_cells_on_model_levels: np.ndarray,
        ddz_of_reference_exner_at_cells_on_half_levels: np.ndarray,
        ddqz_z_half: np.ndarray,
        nonhydro_buoy_at_cells_on_half_levels: np.ndarray,
        rho_at_cells_on_half_levels: np.ndarray,
        exner_at_cells_on_half_levels: np.ndarray,
        time_extrapolation_parameter_for_exner: np.ndarray,
        current_exner: np.ndarray,
        reference_exner_at_cells_on_model_levels: np.ndarray,
        temporal_extrapolation_of_perturbed_exner: np.ndarray,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: np.ndarray,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: np.ndarray,
        theta_v_at_cells_on_half_levels: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        d2dexdz2_fac1_mc: np.ndarray,
        d2dexdz2_fac2_mc: np.ndarray,
        igradp_method: gtx.int32,
        nflatlev: gtx.int32,
        nflat_gradp: gtx.int32,
        start_cell_lateral_boundary: gtx.int32,
        start_cell_lateral_boundary_level_3: gtx.int32,
        start_cell_halo_level_2: gtx.int32,
        end_cell_halo: gtx.int32,
        end_cell_halo_level_2: gtx.int32,
        **kwargs: Any,
    ) -> dict:
        vert_idx = np.arange(kwargs["surface_level"])
        cell = np.arange(end_cell_halo_level_2)
        horz_idx = cell[:, np.newaxis]
        surface_level = kwargs["surface_level"]

        (
            perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels[:, : surface_level - 1],
        ) = np.where(
            (start_cell_lateral_boundary <= horz_idx)
            & (horz_idx < start_cell_lateral_boundary_level_3),
            (
                np.zeros_like(perturbed_rho_at_cells_on_model_levels),
                np.zeros_like(perturbed_theta_v_at_cells_on_model_levels[:, : surface_level - 1]),
            ),
            (
                perturbed_rho_at_cells_on_model_levels,
                perturbed_theta_v_at_cells_on_model_levels[:, : surface_level - 1],
            ),
        )

        (
            temporal_extrapolation_of_perturbed_exner[:, : surface_level - 1],
            perturbed_exner_at_cells_on_model_levels,
        ) = np.where(
            (start_cell_lateral_boundary_level_3 <= horz_idx) & (horz_idx < end_cell_halo),
            extrapolate_temporally_exner_pressure_numpy(
                connectivities=connectivities,
                exner=current_exner,
                exner_ref_mc=reference_exner_at_cells_on_model_levels,
                exner_pr=perturbed_exner_at_cells_on_model_levels,
                exner_exfac=time_extrapolation_parameter_for_exner,
            ),
            (
                temporal_extrapolation_of_perturbed_exner[:, : surface_level - 1],
                perturbed_exner_at_cells_on_model_levels,
            ),
        )

        temporal_extrapolation_of_perturbed_exner = np.where(
            (start_cell_lateral_boundary_level_3 <= horz_idx)
            & (horz_idx < end_cell_halo)
            & (vert_idx == surface_level - 1),
            np.zeros_like(temporal_extrapolation_of_perturbed_exner),
            temporal_extrapolation_of_perturbed_exner,
        )
        if igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            exner_at_cells_on_half_levels = np.where(
                (start_cell_lateral_boundary_level_3 <= horz_idx)
                & (horz_idx < end_cell_halo)
                & (vert_idx == surface_level - 1),
                interpolate_to_surface_numpy(
                    interpolant=temporal_extrapolation_of_perturbed_exner,
                    wgtfacq_c=wgtfacq_c,
                    interpolation_to_surface=exner_at_cells_on_half_levels,
                ),
                exner_at_cells_on_half_levels,
            )
            exner_at_cells_on_half_levels = np.where(
                (start_cell_lateral_boundary_level_3 <= horz_idx)
                & (horz_idx < end_cell_halo)
                & (max(1, nflatlev) <= vert_idx)
                & (vert_idx < surface_level - 1),
                interpolate_cell_field_to_half_levels_vp_numpy(
                    wgtfac_c=wgtfac_c, interpolant=temporal_extrapolation_of_perturbed_exner
                ),
                exner_at_cells_on_half_levels,
            )

            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = np.where(
                (start_cell_lateral_boundary_level_3 <= horz_idx)
                & (horz_idx < end_cell_halo)
                & (nflatlev <= vert_idx[: surface_level - 1]),
                compute_first_vertical_derivative_numpy(
                    cell_kdim_field=exner_at_cells_on_half_levels, inv_ddqz_z_full=inv_ddqz_z_full
                ),
                ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            )

        (
            perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels[:, : surface_level - 1],
        ) = np.where(
            (start_cell_lateral_boundary_level_3 <= horz_idx)
            & (horz_idx < end_cell_halo)
            & (vert_idx[: surface_level - 1] == gtx.int32(0)),
            compute_perturbation_of_rho_and_theta_numpy(
                rho=current_rho,
                rho_ref_mc=reference_rho_at_cells_on_model_levels,
                theta_v=current_theta_v,
                theta_ref_mc=reference_theta_at_cells_on_model_levels,
            ),
            (
                perturbed_rho_at_cells_on_model_levels,
                perturbed_theta_v_at_cells_on_model_levels[:, : surface_level - 1],
            ),
        )

        (
            rho_at_cells_on_half_levels,
            perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels[:, : surface_level - 1],
        ) = np.where(
            (start_cell_lateral_boundary_level_3 <= horz_idx)
            & (horz_idx < end_cell_halo)
            & (vert_idx[: surface_level - 1] >= gtx.int32(1)),
            compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers_numpy(
                wgtfac_c=wgtfac_c[:, : surface_level - 1],
                rho=current_rho,
                rho_ref_mc=reference_rho_at_cells_on_model_levels,
                theta_v=current_theta_v,
                theta_ref_mc=reference_theta_at_cells_on_model_levels,
            ),
            (
                rho_at_cells_on_half_levels,
                perturbed_rho_at_cells_on_model_levels,
                perturbed_theta_v_at_cells_on_model_levels[:, : surface_level - 1],
            ),
        )

        (
            perturbed_theta_v_at_cells_on_half_levels[:, : surface_level - 1],
            theta_v_at_cells_on_half_levels[:, : surface_level - 1],
            nonhydro_buoy_at_cells_on_half_levels,
        ) = np.where(
            (start_cell_lateral_boundary_level_3 <= horz_idx)
            & (horz_idx < end_cell_halo)
            & (vert_idx[: surface_level - 1] >= gtx.int32(1)),
            compute_virtual_potential_temperatures_and_pressure_gradient_numpy(
                connectivities=connectivities,
                wgtfac_c=wgtfac_c[:, : surface_level - 1],
                z_rth_pr_2=perturbed_theta_v_at_cells_on_model_levels[:, : surface_level - 1],
                theta_v=current_theta_v,
                vwind_expl_wgt=exner_w_explicit_weight_parameter,
                exner_pr=perturbed_exner_at_cells_on_model_levels,
                d_exner_dz_ref_ic=ddz_of_reference_exner_at_cells_on_half_levels,
                ddqz_z_half=ddqz_z_half,
            ),
            (
                perturbed_theta_v_at_cells_on_half_levels[:, : surface_level - 1],
                theta_v_at_cells_on_half_levels[:, : surface_level - 1],
                nonhydro_buoy_at_cells_on_half_levels,
            ),
        )

        (perturbed_theta_v_at_cells_on_half_levels, theta_v_at_cells_on_half_levels) = np.where(
            (vert_idx == surface_level - 1)
            & (start_cell_lateral_boundary_level_3 <= horz_idx)
            & (horz_idx < end_cell_halo),
            set_theta_v_prime_ic_at_lower_boundary_numpy(
                wgtfacq_c=wgtfacq_c,
                z_rth_pr=perturbed_theta_v_at_cells_on_model_levels,
                theta_ref_ic=reference_theta_at_cells_on_half_levels,
                z_theta_v_pr_ic=np.zeros_like(perturbed_theta_v_at_cells_on_half_levels),
                theta_v_ic=np.zeros_like(theta_v_at_cells_on_half_levels),
            ),
            (perturbed_theta_v_at_cells_on_half_levels, theta_v_at_cells_on_half_levels),
        )
        if igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = np.where(
                (start_cell_lateral_boundary_level_3 <= horz_idx)
                & (horz_idx < end_cell_halo)
                & (nflat_gradp <= vert_idx[: surface_level - 1]),
                compute_approx_of_2nd_vertical_derivative_of_exner_numpy(
                    z_theta_v_pr_ic=perturbed_theta_v_at_cells_on_half_levels,
                    d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
                    d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
                    z_rth_pr_2=perturbed_theta_v_at_cells_on_model_levels[:, : surface_level - 1],
                ),
                d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            )

        (
            perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels[:, : surface_level - 1],
        ) = np.where(
            (start_cell_halo_level_2 <= horz_idx) & (horz_idx < end_cell_halo_level_2),
            compute_perturbation_of_rho_and_theta_numpy(
                rho=current_rho,
                rho_ref_mc=reference_rho_at_cells_on_model_levels,
                theta_v=current_theta_v,
                theta_ref_mc=reference_theta_at_cells_on_model_levels,
            ),
            (
                perturbed_rho_at_cells_on_model_levels,
                perturbed_theta_v_at_cells_on_model_levels[:, : surface_level - 1],
            ),
        )

        return dict(
            temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
            exner_at_cells_on_half_levels=exner_at_cells_on_half_levels,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            perturbed_rho_at_cells_on_model_levels=perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v_at_cells_on_model_levels,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels=perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            nonhydro_buoy_at_cells_on_half_levels=nonhydro_buoy_at_cells_on_half_levels,
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        reference_rho_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        reference_theta_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        wgtfacq_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        perturbed_rho_at_cells_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim
        )
        perturbed_theta_v_at_cells_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        perturbed_theta_v_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        reference_theta_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        d2dexdz2_fac1_mc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        d2dexdz2_fac2_mc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        wgtfac_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        exner_w_explicit_weight_parameter = data_alloc.random_field(grid, dims.CellDim)
        perturbed_exner_at_cells_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim
        )
        ddz_of_reference_exner_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        ddqz_z_half = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        nonhydro_buoy_at_cells_on_half_levels = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        rho_at_cells_on_half_levels = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        exner_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        time_extrapolation_parameter_for_exner = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        current_exner = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        reference_exner_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        temporal_extrapolation_of_perturbed_exner = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim
        )
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim
        )
        theta_v_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        inv_ddqz_z_full = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        current_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        current_theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim)

        igradp_method = HorizontalPressureDiscretizationType.TAYLOR_HYDRO

        cell_domain = h_grid.domain(dims.CellDim)
        start_cell_lateral_boundary = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY))
        start_cell_lateral_boundary_level_3 = grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        end_cell_halo = grid.end_index(cell_domain(h_grid.Zone.HALO))
        start_cell_halo_level_2 = grid.start_index(cell_domain(h_grid.Zone.HALO_LEVEL_2))
        end_cell_halo_level_2 = grid.end_index(cell_domain(h_grid.Zone.HALO_LEVEL_2))

        nflatlev = 5  # value is set to reflect the MCH ch1 experiment. Changing this value will change the expected runtime
        nflat_gradp = 34  # value is set to reflect the MCH ch1 experiment. Changing this value will change the expected runtime

        return dict(
            temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
            exner_at_cells_on_half_levels=exner_at_cells_on_half_levels,
            perturbed_rho_at_cells_on_model_levels=perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v_at_cells_on_model_levels,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels=perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            current_rho=current_rho,
            reference_rho_at_cells_on_model_levels=reference_rho_at_cells_on_model_levels,
            current_theta_v=current_theta_v,
            reference_theta_at_cells_on_model_levels=reference_theta_at_cells_on_model_levels,
            reference_theta_at_cells_on_half_levels=reference_theta_at_cells_on_half_levels,
            wgtfacq_c=wgtfacq_c,
            wgtfac_c=wgtfac_c,
            exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
            ddz_of_reference_exner_at_cells_on_half_levels=ddz_of_reference_exner_at_cells_on_half_levels,
            ddqz_z_half=ddqz_z_half,
            nonhydro_buoy_at_cells_on_half_levels=nonhydro_buoy_at_cells_on_half_levels,
            time_extrapolation_parameter_for_exner=time_extrapolation_parameter_for_exner,
            current_exner=current_exner,
            reference_exner_at_cells_on_model_levels=reference_exner_at_cells_on_model_levels,
            inv_ddqz_z_full=inv_ddqz_z_full,
            d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
            d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
            igradp_method=igradp_method,
            nflatlev=nflatlev,
            nflat_gradp=nflat_gradp,
            start_cell_lateral_boundary=start_cell_lateral_boundary,
            start_cell_lateral_boundary_level_3=start_cell_lateral_boundary_level_3,
            start_cell_halo_level_2=start_cell_halo_level_2,
            end_cell_halo=end_cell_halo,
            end_cell_halo_level_2=end_cell_halo_level_2,
            model_top=0,
            surface_level=grid.num_levels + 1,
        )

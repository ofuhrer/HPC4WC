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

from icon4py.model.atmosphere.dycore.stencils.compute_cell_diagnostics_for_dycore import (
    compute_interpolation_and_nonhydro_buoy,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


@pytest.mark.continuous_benchmarking
class TestInterpolateRhoThetaVToHalfLevelsAndComputePressureBuoyancyAcceleration(
    stencil_tests.StencilTest
):
    PROGRAM = compute_interpolation_and_nonhydro_buoy
    OUTPUTS = (
        "rho_at_cells_on_half_levels",
        "perturbed_theta_v_at_cells_on_half_levels",
        "theta_v_at_cells_on_half_levels",
        "nonhydro_buoy_at_cells_on_half_levels",
    )
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        rho_at_cells_on_half_levels: np.ndarray,
        perturbed_theta_v_at_cells_on_half_levels: np.ndarray,
        theta_v_at_cells_on_half_levels: np.ndarray,
        nonhydro_buoy_at_cells_on_half_levels: np.ndarray,
        w: np.ndarray,
        contravariant_correction_at_cells_on_half_levels: np.ndarray,
        current_rho: np.ndarray,
        next_rho: np.ndarray,
        current_theta_v: np.ndarray,
        next_theta_v: np.ndarray,
        perturbed_exner_at_cells_on_model_levels: np.ndarray,
        reference_theta_at_cells_on_model_levels: np.ndarray,
        ddz_of_reference_exner_at_cells_on_half_levels: np.ndarray,
        ddqz_z_half: np.ndarray,
        wgtfac_c: np.ndarray,
        exner_w_explicit_weight_parameter: np.ndarray,
        dtime: ta.wpfloat,
        rhotheta_explicit_weight_parameter: ta.wpfloat,
        rhotheta_implicit_weight_parameter: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        exner_w_explicit_weight_parameter = np.expand_dims(
            exner_w_explicit_weight_parameter, axis=-1
        )
        koffset_current_rho = np.roll(current_rho, shift=1, axis=1)
        koffset_next_rho = np.roll(next_rho, shift=1, axis=1)
        koffset_current_theta_v = np.roll(current_theta_v, shift=1, axis=1)
        koffset_next_theta_v = np.roll(next_theta_v, shift=1, axis=1)
        koffset_reference_theta_at_cells_on_model_levels = np.roll(
            reference_theta_at_cells_on_model_levels, shift=1, axis=1
        )
        koffset_perturbed_exner_at_cells_on_model_levels = np.roll(
            perturbed_exner_at_cells_on_model_levels, shift=1, axis=1
        )

        back_trajectory_w_at_cells_on_half_levels = (
            -(w - contravariant_correction_at_cells_on_half_levels) * dtime * 0.5 / ddqz_z_half
        )
        time_averaged_rho_kup = (
            rhotheta_explicit_weight_parameter * koffset_current_rho
            + rhotheta_implicit_weight_parameter * koffset_next_rho
        )
        time_averaged_theta_v_kup = (
            rhotheta_explicit_weight_parameter * koffset_current_theta_v
            + rhotheta_implicit_weight_parameter * koffset_next_theta_v
        )
        time_averaged_rho = (
            rhotheta_explicit_weight_parameter * current_rho
            + rhotheta_implicit_weight_parameter * next_rho
        )
        time_averaged_theta_v = (
            rhotheta_explicit_weight_parameter * current_theta_v
            + rhotheta_implicit_weight_parameter * next_theta_v
        )
        rho_at_cells_on_half_levels_full = (
            wgtfac_c * time_averaged_rho
            + (1 - wgtfac_c) * time_averaged_rho_kup
            + back_trajectory_w_at_cells_on_half_levels
            * (time_averaged_rho_kup - time_averaged_rho)
        )
        time_averaged_perturbed_theta_v_kup = (
            time_averaged_theta_v_kup - koffset_reference_theta_at_cells_on_model_levels
        )
        time_averaged_perturbed_theta_v = (
            time_averaged_theta_v - reference_theta_at_cells_on_model_levels
        )
        perturbed_theta_v_at_cells_on_half_levels_full = (
            wgtfac_c * time_averaged_perturbed_theta_v
            + (1 - wgtfac_c) * time_averaged_perturbed_theta_v_kup
        )
        theta_v_at_cells_on_half_levels_full = (
            wgtfac_c * time_averaged_theta_v
            + (1 - wgtfac_c) * time_averaged_theta_v_kup
            + back_trajectory_w_at_cells_on_half_levels
            * (time_averaged_theta_v_kup - time_averaged_theta_v)
        )
        nonhydro_buoy_at_cells_on_half_levels_full = (
            exner_w_explicit_weight_parameter
            * theta_v_at_cells_on_half_levels_full
            * (
                koffset_perturbed_exner_at_cells_on_model_levels
                - perturbed_exner_at_cells_on_model_levels
            )
            / ddqz_z_half
            + perturbed_theta_v_at_cells_on_half_levels_full
            * ddz_of_reference_exner_at_cells_on_half_levels
        )

        horizontal_start = kwargs["horizontal_start"]
        horizontal_end = kwargs["horizontal_end"]
        vertical_start = kwargs["vertical_start"]
        vertical_end = kwargs["vertical_end"]

        rho_at_cells_on_half_levels[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ] = rho_at_cells_on_half_levels_full[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        perturbed_theta_v_at_cells_on_half_levels[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ] = perturbed_theta_v_at_cells_on_half_levels_full[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        theta_v_at_cells_on_half_levels[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ] = theta_v_at_cells_on_half_levels_full[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        nonhydro_buoy_at_cells_on_half_levels[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ] = nonhydro_buoy_at_cells_on_half_levels_full[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        return dict(
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels=perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            nonhydro_buoy_at_cells_on_half_levels=nonhydro_buoy_at_cells_on_half_levels,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        contravariant_correction_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        ddqz_z_half = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        current_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        next_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        current_theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        next_theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        wgtfac_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        reference_theta_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        exner_w_explicit_weight_parameter = data_alloc.random_field(grid, dims.CellDim)
        perturbed_exner_at_cells_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim
        )
        ddz_of_reference_exner_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        rho_at_cells_on_half_levels = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        perturbed_theta_v_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        theta_v_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        nonhydro_buoy_at_cells_on_half_levels = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)

        dtime = 0.9
        rhotheta_explicit_weight_parameter = 0.25
        rhotheta_implicit_weight_parameter = 0.75

        cell_domain = h_grid.domain(dims.CellDim)
        start_cell_lateral_boundary_level_3 = grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        end_cell_local = grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        return dict(
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels=perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            nonhydro_buoy_at_cells_on_half_levels=nonhydro_buoy_at_cells_on_half_levels,
            w=w,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            current_rho=current_rho,
            next_rho=next_rho,
            current_theta_v=current_theta_v,
            next_theta_v=next_theta_v,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
            reference_theta_at_cells_on_model_levels=reference_theta_at_cells_on_model_levels,
            ddz_of_reference_exner_at_cells_on_half_levels=ddz_of_reference_exner_at_cells_on_half_levels,
            ddqz_z_half=ddqz_z_half,
            wgtfac_c=wgtfac_c,
            exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
            dtime=dtime,
            rhotheta_explicit_weight_parameter=rhotheta_explicit_weight_parameter,
            rhotheta_implicit_weight_parameter=rhotheta_implicit_weight_parameter,
            horizontal_start=start_cell_lateral_boundary_level_3,
            horizontal_end=end_cell_local,
            vertical_start=1,
            vertical_end=grid.num_levels,
        )

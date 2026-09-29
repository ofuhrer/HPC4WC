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
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence import (
    apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing import definitions
from icon4py.model.testing.stencil_tests import StandardStaticVariants, StencilTest

from .test_apply_nabla2_to_w import apply_nabla2_to_w_numpy
from .test_apply_nabla2_to_w_in_upper_damping_layer import (
    apply_nabla2_to_w_in_upper_damping_layer_numpy,
)
from .test_calculate_horizontal_gradients_for_turbulence import (
    calculate_horizontal_gradients_for_turbulence_numpy,
)
from .test_calculate_nabla2_for_w import calculate_nabla2_for_w_numpy


@pytest.mark.embedded_remap_error
@pytest.mark.continuous_benchmarking
class TestApplyDiffusionToWAndComputeHorizontalGradientsForTurbulence(StencilTest):
    PROGRAM = apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence
    OUTPUTS = ("w", "dwdx", "dwdy")
    STATIC_PARAMS = {
        StandardStaticVariants.NONE: (),
        StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "halo_idx",
            "interior_idx",
            "vertical_start",
            "vertical_end",
            "nrdmax",
            "type_shear",
        ),
        StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "nrdmax",
            "type_shear",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        area,
        geofac_n2s,
        geofac_grg_x,
        geofac_grg_y,
        w_old,
        type_shear,
        dwdx,
        dwdy,
        diff_multfac_w,
        diff_multfac_n2w,
        nrdmax,
        interior_idx,
        halo_idx,
        horizontal_start,
        horizontal_end,
        vertical_start,
        vertical_end,
        **kwargs,
    ) -> dict:
        k = np.arange(w_old.shape[1])
        cell = np.arange(w_old.shape[0])
        reshaped_k = k[np.newaxis, :]
        reshaped_cell = cell[:, np.newaxis]
        out_w, out_dwdx, out_dwdy = (
            np.zeros_like(w_old),
            dwdx.copy(),
            dwdy.copy(),
        )  # create output arrays to update only the necessary slices
        if type_shear == 2:
            dwdx, dwdy = np.where(
                reshaped_k > 0,
                calculate_horizontal_gradients_for_turbulence_numpy(
                    connectivities, w_old, geofac_grg_x, geofac_grg_y
                ),
                (dwdx, dwdy),
            )

        z_nabla2_c = calculate_nabla2_for_w_numpy(connectivities, w_old, geofac_n2s)

        w = np.where(
            (interior_idx <= reshaped_cell) & (reshaped_cell < halo_idx),
            apply_nabla2_to_w_numpy(
                connectivities=connectivities,
                area=area,
                z_nabla2_c=z_nabla2_c,
                geofac_n2s=geofac_n2s,
                w=w_old,
                diff_multfac_w=diff_multfac_w,
            ),
            w_old,
        )

        w = np.where(
            (reshaped_k > 0)
            & (reshaped_k < nrdmax)
            & (interior_idx <= reshaped_cell)
            & (reshaped_cell < halo_idx),
            apply_nabla2_to_w_in_upper_damping_layer_numpy(w, diff_multfac_n2w, area, z_nabla2_c),
            w,
        )
        subset = (slice(horizontal_start, horizontal_end), slice(vertical_start, vertical_end))
        out_w[subset] = w[subset]
        out_dwdx[subset] = dwdx[subset]
        out_dwdy[subset] = dwdy[subset]
        return dict(w=out_w, dwdx=out_dwdx, dwdy=out_dwdy)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        nrdmax = 13
        cell_domain = h_grid.domain(dims.CellDim)
        interior_idx = grid.start_index(cell_domain(h_grid.Zone.INTERIOR))  # 0 for simple grid
        halo_idx = grid.end_index(
            cell_domain(h_grid.Zone.LOCAL)
        )  # same as horizontal_end for simple grid
        type_shear = 2

        def _get_start_index_for_w_diffusion() -> int32:
            return (
                grid.start_index(cell_domain(h_grid.Zone.NUDGING))
                if grid.limited_area
                else grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
            )

        horizontal_start = _get_start_index_for_w_diffusion()
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.HALO))

        geofac_grg_x = random_field(grid, dims.CellDim, dims.C2E2CODim)
        geofac_grg_y = random_field(grid, dims.CellDim, dims.C2E2CODim)
        diff_multfac_n2w = random_field(grid, dims.KDim)
        area = random_field(grid, dims.CellDim)
        geofac_n2s = random_field(grid, dims.CellDim, dims.C2E2CODim)
        w_old = random_field(grid, dims.CellDim, dims.KDim)
        diff_multfac_w = 5.0

        w = zero_field(grid, dims.CellDim, dims.KDim)
        dwdx = random_field(grid, dims.CellDim, dims.KDim)
        dwdy = random_field(grid, dims.CellDim, dims.KDim)

        return dict(
            area=area,
            geofac_n2s=geofac_n2s,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            w_old=w_old,
            type_shear=type_shear,
            diff_multfac_w=diff_multfac_w,
            diff_multfac_n2w=diff_multfac_n2w,
            nrdmax=nrdmax,
            interior_idx=interior_idx,
            halo_idx=halo_idx,
            w=w,
            dwdx=dwdx,
            dwdy=dwdy,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )

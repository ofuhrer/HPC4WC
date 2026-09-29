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

from icon4py.model.atmosphere.dycore.stencils.compute_advection_in_vertical_momentum_equation import (
    compute_advection_in_corrector_vertical_momentum,
    compute_advection_in_predictor_vertical_momentum,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests

from .test_add_interpolated_horizontal_advection_of_w import (
    add_interpolated_horizontal_advection_of_w_numpy,
)
from .test_compute_advective_vertical_wind_tendency import (
    compute_advective_vertical_wind_tendency_numpy,
)
from .test_compute_horizontal_advection_term_for_vertical_velocity import (
    compute_horizontal_advection_term_for_vertical_velocity_numpy,
)
from .test_interpolate_cell_field_to_half_levels_vp import (
    interpolate_cell_field_to_half_levels_vp_numpy,
)
from .test_interpolate_to_cell_center import interpolate_to_cell_center_numpy
from .test_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy,
)


def interpolate_contravariant_correction_to_cells_on_half_levels_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    contravariant_correction_at_cells_on_half_levels: np.ndarray,
    contravariant_correction_at_edges_on_model_levels: np.ndarray,
    e_bln_c_s: np.ndarray,
    wgtfac_c: np.ndarray,
    nflatlev: int,
    nlev: int,
) -> np.ndarray:
    k = np.arange(nlev)

    contravariant_correction_at_cells_model_levels = interpolate_to_cell_center_numpy(
        connectivities, contravariant_correction_at_edges_on_model_levels, e_bln_c_s
    )

    condition = k >= nflatlev + 1
    contravariant_correction_at_cells_on_half_levels = np.where(
        condition,
        interpolate_cell_field_to_half_levels_vp_numpy(
            wgtfac_c=wgtfac_c, interpolant=contravariant_correction_at_cells_model_levels
        ),
        np.zeros_like(contravariant_correction_at_cells_on_half_levels),
    )

    return contravariant_correction_at_cells_on_half_levels


def interpolate_contravariant_vertical_velocity_to_full_levels_numpy(
    contravariant_corrected_w_at_cells_on_half_levels: np.ndarray,
) -> np.ndarray:
    num_rows, num_cols = contravariant_corrected_w_at_cells_on_half_levels.shape
    contravariant_corrected_w_with_surface = np.zeros((num_rows, num_cols + 1))
    contravariant_corrected_w_with_surface[:, :-1] = (
        contravariant_corrected_w_at_cells_on_half_levels
    )
    contravariant_corrected_w_at_cells_on_model_levels = 0.5 * (
        contravariant_corrected_w_with_surface[:, :-1]
        + contravariant_corrected_w_with_surface[:, 1:]
    )
    return contravariant_corrected_w_at_cells_on_model_levels


def compute_maximum_cfl_and_clip_contravariant_vertical_velocity_numpy(
    *,
    w: np.ndarray,
    contravariant_correction_at_cells_on_half_levels: np.ndarray,
    ddqz_z_half: np.ndarray,
    cfl_w_limit: ta.wpfloat,
    dtime: ta.wpfloat,
    nlev: int,
    end_index_of_damping_layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_rows, num_cols = contravariant_correction_at_cells_on_half_levels.shape

    k = np.arange(num_cols)
    condition = (np.maximum(2, end_index_of_damping_layer - 2) <= k) & (k < nlev - 3)

    contravariant_corrected_w_at_cells_on_half_levels = (
        w - contravariant_correction_at_cells_on_half_levels
    )

    cfl_clipping = np.where(
        (np.abs(contravariant_corrected_w_at_cells_on_half_levels) > cfl_w_limit * ddqz_z_half)
        & condition,
        np.ones([num_rows, num_cols]),
        np.zeros_like(contravariant_corrected_w_at_cells_on_half_levels),
    )
    vertical_cfl = np.where(
        cfl_clipping == 1.0,
        contravariant_corrected_w_at_cells_on_half_levels * dtime / ddqz_z_half,
        0.0,
    )
    contravariant_corrected_w_at_cells_on_half_levels = np.where(
        (cfl_clipping == 1.0) & (vertical_cfl < -0.85),
        -0.85 * ddqz_z_half / dtime,
        contravariant_corrected_w_at_cells_on_half_levels,
    )
    contravariant_corrected_w_at_cells_on_half_levels = np.where(
        (cfl_clipping == 1.0) & (vertical_cfl > 0.85),
        0.85 * ddqz_z_half / dtime,
        contravariant_corrected_w_at_cells_on_half_levels,
    )

    return contravariant_corrected_w_at_cells_on_half_levels, cfl_clipping, vertical_cfl


def compute_horizontal_advection_of_w(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    w: np.ndarray,
    tangential_wind_on_half_levels: np.ndarray,
    vn_on_half_levels: np.ndarray,
    c_intp: np.ndarray,
    inv_dual_edge_length: np.ndarray,
    inv_primal_edge_length: np.ndarray,
    tangent_orientation: np.ndarray,
) -> np.ndarray:
    w_at_vertices = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(
        connectivities, w, c_intp
    )

    horizontal_advection_of_w_at_edges_on_half_levels = (
        compute_horizontal_advection_term_for_vertical_velocity_numpy(
            connectivities=connectivities,
            vn_ie=vn_on_half_levels,
            inv_dual_edge_length=inv_dual_edge_length,
            w=w,
            z_vt_ie=tangential_wind_on_half_levels,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            z_w_v=w_at_vertices,
        )
    )

    return horizontal_advection_of_w_at_edges_on_half_levels


def add_extra_diffusion_for_w_approaching_cfl_wihtout_levmask_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    cfl_clipping: np.ndarray,
    owner_mask: np.ndarray,
    contravariant_corrected_w_at_cells_on_half_levels: np.ndarray,
    ddqz_z_half: np.ndarray,
    area: np.ndarray,
    geofac_n2s: np.ndarray,
    w: np.ndarray,
    vertical_wind_advective_tendency: np.ndarray,
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.wpfloat,
    dtime: ta.wpfloat,
) -> np.ndarray:
    owner_mask = np.expand_dims(owner_mask, axis=-1)
    area = np.expand_dims(area, axis=-1)
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)

    difcoef = np.where(
        (cfl_clipping == 1) & (owner_mask == 1),
        scalfac_exdiff
        * np.minimum(
            0.85 - cfl_w_limit * dtime,
            np.abs(contravariant_corrected_w_at_cells_on_half_levels) * dtime / ddqz_z_half
            - cfl_w_limit * dtime,
        ),
        0,
    )

    c2e2cO = connectivities[dims.C2E2CODim]
    vertical_wind_advective_tendency = np.where(
        (cfl_clipping == 1) & (owner_mask == 1),
        vertical_wind_advective_tendency
        + difcoef
        * area
        * np.sum(
            np.where(
                (c2e2cO != -1)[:, :, np.newaxis],
                w[c2e2cO] * geofac_n2s,
                0,
            ),
            axis=1,
        ),
        vertical_wind_advective_tendency,
    )
    return vertical_wind_advective_tendency


def compute_advective_vertical_wind_tendency_and_apply_diffusion_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    vertical_wind_advective_tendency: np.ndarray,
    w: np.ndarray,
    horizontal_advection_of_w_at_edges_on_half_levels: np.ndarray,
    contravariant_corrected_w_at_cells_on_half_levels: np.ndarray,
    cfl_clipping: np.ndarray,
    coeff1_dwdz: np.ndarray,
    coeff2_dwdz: np.ndarray,
    e_bln_c_s: np.ndarray,
    ddqz_z_half: np.ndarray,
    area: np.ndarray,
    geofac_n2s: np.ndarray,
    owner_mask: np.ndarray,
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.wpfloat,
    dtime: ta.wpfloat,
    nlev: int,
    end_index_of_damping_layer: int,
) -> np.ndarray:
    k = np.arange(nlev)

    condition1 = k >= 1
    vertical_wind_advective_tendency = np.where(
        condition1,
        compute_advective_vertical_wind_tendency_numpy(
            contravariant_corrected_w_at_cells_on_half_levels, w, coeff1_dwdz, coeff2_dwdz
        ),
        vertical_wind_advective_tendency,
    )

    vertical_wind_advective_tendency = np.where(
        condition1,
        add_interpolated_horizontal_advection_of_w_numpy(
            connectivities,
            e_bln_c_s,
            horizontal_advection_of_w_at_edges_on_half_levels[:, :-1],
            vertical_wind_advective_tendency,
        ),
        vertical_wind_advective_tendency,
    )

    condition2 = (np.maximum(2, end_index_of_damping_layer - 2) <= k) & (k < nlev - 3)

    vertical_wind_advective_tendency = np.where(
        condition2,
        add_extra_diffusion_for_w_approaching_cfl_wihtout_levmask_numpy(
            connectivities=connectivities,
            cfl_clipping=cfl_clipping,
            owner_mask=owner_mask,
            contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            w=w[:, :-1],
            vertical_wind_advective_tendency=vertical_wind_advective_tendency,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
        ),
        vertical_wind_advective_tendency,
    )

    return vertical_wind_advective_tendency


@pytest.mark.embedded_remap_error
@pytest.mark.continuous_benchmarking
class TestFusedVelocityAdvectionStencilVMomentum(stencil_tests.StencilTest):
    PROGRAM = compute_advection_in_corrector_vertical_momentum
    OUTPUTS = (
        "vertical_wind_advective_tendency",
        "contravariant_corrected_w_at_cells_on_model_levels",
        "vertical_cfl",
    )
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
            "end_index_of_damping_layer",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "end_index_of_damping_layer",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        vertical_wind_advective_tendency: np.ndarray,
        contravariant_corrected_w_at_cells_on_model_levels: np.ndarray,
        vertical_cfl: np.ndarray,
        w: np.ndarray,
        tangential_wind_on_half_levels: np.ndarray,
        vn_on_half_levels: np.ndarray,
        contravariant_correction_at_cells_on_half_levels: np.ndarray,
        coeff1_dwdz: np.ndarray,
        coeff2_dwdz: np.ndarray,
        c_intp: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        tangent_orientation: np.ndarray,
        e_bln_c_s: np.ndarray,
        ddqz_z_half: np.ndarray,
        area: np.ndarray,
        geofac_n2s: np.ndarray,
        scalfac_exdiff: ta.wpfloat,
        cfl_w_limit: ta.wpfloat,
        dtime: ta.wpfloat,
        owner_mask: np.ndarray,
        end_index_of_damping_layer: int,
        **kwargs: Any,
    ) -> dict:
        nlev = kwargs["vertical_end"]

        horizontal_advection_of_w_at_edges_on_half_levels = compute_horizontal_advection_of_w(
            connectivities=connectivities,
            w=w,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=vn_on_half_levels,
            c_intp=c_intp,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
        )

        # We need to store the initial return field, because we only compute on a subdomain.
        contravariant_corrected_w_at_cells_on_model_levels_ret = (
            contravariant_corrected_w_at_cells_on_model_levels.copy()
        )
        vertical_wind_advective_tendency_ret = vertical_wind_advective_tendency.copy()
        vertical_cfl_ret = vertical_cfl.copy()

        (
            contravariant_corrected_w_at_cells_on_half_levels,
            cfl_clipping,
            vertical_cfl,
        ) = compute_maximum_cfl_and_clip_contravariant_vertical_velocity_numpy(
            w=w[:, :-1],
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels[
                :, :-1
            ],
            ddqz_z_half=ddqz_z_half,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            nlev=nlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
        )

        vertical_wind_advective_tendency = compute_advective_vertical_wind_tendency_and_apply_diffusion_numpy(
            connectivities=connectivities,
            vertical_wind_advective_tendency=vertical_wind_advective_tendency,
            w=w,
            horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
            contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
            cfl_clipping=cfl_clipping,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            e_bln_c_s=e_bln_c_s,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            owner_mask=owner_mask,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            nlev=nlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
        )

        contravariant_corrected_w_at_cells_on_model_levels = (
            interpolate_contravariant_vertical_velocity_to_full_levels_numpy(
                contravariant_corrected_w_at_cells_on_half_levels
            )
        )

        # Apply the slicing.
        horizontal_start = kwargs["horizontal_start"]
        horizontal_end = kwargs["horizontal_end"]
        vertical_start = kwargs["vertical_start"]
        vertical_end = kwargs["vertical_end"]

        contravariant_corrected_w_at_cells_on_model_levels_ret[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ] = contravariant_corrected_w_at_cells_on_model_levels[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        vertical_wind_advective_tendency_ret[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ] = vertical_wind_advective_tendency[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        vertical_cfl_ret[horizontal_start:horizontal_end, vertical_start:vertical_end] = (
            vertical_cfl[horizontal_start:horizontal_end, vertical_start:vertical_end]
        )

        return dict(
            vertical_wind_advective_tendency=vertical_wind_advective_tendency_ret,
            contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels_ret,
            vertical_cfl=vertical_cfl_ret,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        contravariant_corrected_w_at_cells_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim
        )
        vertical_wind_advective_tendency = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        tangential_wind_on_half_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}
        )
        vn_on_half_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}
        )
        contravariant_correction_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )

        coeff1_dwdz = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        coeff2_dwdz = data_alloc.random_field(grid, dims.CellDim, dims.KDim)

        c_intp = data_alloc.random_field(grid, dims.VertexDim, dims.V2CDim)
        inv_dual_edge_length = data_alloc.random_field(grid, dims.EdgeDim, low=1.0e-5)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim, low=1.0e-5)
        tangent_orientation = data_alloc.random_field(grid, dims.EdgeDim, low=1.0e-5)
        e_bln_c_s = data_alloc.random_field(grid, dims.CellDim, dims.C2EDim)

        vertical_cfl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        owner_mask = data_alloc.random_mask(grid, dims.CellDim)
        ddqz_z_half = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        area = data_alloc.random_field(grid, dims.CellDim)
        geofac_n2s = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CODim)

        scalfac_exdiff = 10.0
        dtime = 2.0
        cfl_w_limit = 0.65 / dtime

        end_index_of_damping_layer = 12  # value is set to reflect the MCH ch1 experiment. Changing this value will change the expected runtime

        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.HALO))
        vertical_start = 0
        vertical_end = grid.num_levels

        return dict(
            vertical_wind_advective_tendency=vertical_wind_advective_tendency,
            contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
            vertical_cfl=vertical_cfl,
            w=w,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=vn_on_half_levels,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            c_intp=c_intp,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            e_bln_c_s=e_bln_c_s,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            owner_mask=owner_mask,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            end_index_of_damping_layer=end_index_of_damping_layer,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )


@pytest.mark.embedded_remap_error
@pytest.mark.continuous_benchmarking
class TestFusedVelocityAdvectionStencilVMomentumAndContravariant(stencil_tests.StencilTest):
    PROGRAM = compute_advection_in_predictor_vertical_momentum
    OUTPUTS = (
        "contravariant_correction_at_cells_on_half_levels",
        "vertical_wind_advective_tendency",
        "contravariant_corrected_w_at_cells_on_model_levels",
        "vertical_cfl",
    )
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "nflatlev",
            "skip_compute_predictor_vertical_advection",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
            "end_index_of_damping_layer",
            "nflatlev",
            "skip_compute_predictor_vertical_advection",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        contravariant_correction_at_cells_on_half_levels: np.ndarray,
        vertical_wind_advective_tendency: np.ndarray,
        contravariant_corrected_w_at_cells_on_model_levels: np.ndarray,
        vertical_cfl: np.ndarray,
        w: np.ndarray,
        horizontal_advection_of_w_at_edges_on_half_levels: np.ndarray,
        contravariant_correction_at_edges_on_model_levels: np.ndarray,
        coeff1_dwdz: np.ndarray,
        coeff2_dwdz: np.ndarray,
        e_bln_c_s: np.ndarray,
        wgtfac_c: np.ndarray,
        ddqz_z_half: np.ndarray,
        area: np.ndarray,
        geofac_n2s: np.ndarray,
        scalfac_exdiff: ta.wpfloat,
        cfl_w_limit: ta.wpfloat,
        dtime: ta.wpfloat,
        owner_mask: np.ndarray,
        nflatlev: int,
        end_index_of_damping_layer: int,
        skip_compute_predictor_vertical_advection: bool,
        **kwargs: Any,
    ) -> dict:
        nlev = kwargs["vertical_end"]

        # We need to store the initial return field, because we only compute on a subdomain.
        contravariant_correction_at_cells_on_half_levels_ret = (
            contravariant_correction_at_cells_on_half_levels.copy()
        )
        contravariant_corrected_w_at_cells_on_model_levels_ret = (
            contravariant_corrected_w_at_cells_on_model_levels.copy()
        )
        vertical_wind_advective_tendency_ret = vertical_wind_advective_tendency.copy()
        vertical_cfl_ret = vertical_cfl.copy()

        contravariant_correction_at_cells_on_half_levels_nlev = interpolate_contravariant_correction_to_cells_on_half_levels_numpy(
            connectivities=connectivities,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels[
                :, :-1
            ],
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            nflatlev=nflatlev,
            nlev=nlev,
        )

        (
            contravariant_corrected_w_at_cells_on_half_levels,
            cfl_clipping,
            vertical_cfl,
        ) = compute_maximum_cfl_and_clip_contravariant_vertical_velocity_numpy(
            w=w[:, :-1],
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels_nlev,
            ddqz_z_half=ddqz_z_half,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            nlev=nlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
        )
        if not skip_compute_predictor_vertical_advection:
            vertical_wind_advective_tendency = compute_advective_vertical_wind_tendency_and_apply_diffusion_numpy(
                connectivities=connectivities,
                vertical_wind_advective_tendency=vertical_wind_advective_tendency,
                w=w,
                horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
                contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
                cfl_clipping=cfl_clipping,
                coeff1_dwdz=coeff1_dwdz,
                coeff2_dwdz=coeff2_dwdz,
                e_bln_c_s=e_bln_c_s,
                ddqz_z_half=ddqz_z_half,
                area=area,
                geofac_n2s=geofac_n2s,
                owner_mask=owner_mask,
                scalfac_exdiff=scalfac_exdiff,
                cfl_w_limit=cfl_w_limit,
                dtime=dtime,
                nlev=nlev,
                end_index_of_damping_layer=end_index_of_damping_layer,
            )

        contravariant_corrected_w_at_cells_on_model_levels = (
            interpolate_contravariant_vertical_velocity_to_full_levels_numpy(
                contravariant_corrected_w_at_cells_on_half_levels
            )
        )

        # Apply the slicing.
        horizontal_start = kwargs["horizontal_start"]
        horizontal_end = kwargs["horizontal_end"]
        vertical_start = kwargs["vertical_start"]
        vertical_end = kwargs["vertical_end"]

        contravariant_correction_at_cells_on_half_levels_ret[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ] = contravariant_correction_at_cells_on_half_levels_nlev[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        contravariant_corrected_w_at_cells_on_model_levels_ret[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ] = contravariant_corrected_w_at_cells_on_model_levels[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        vertical_wind_advective_tendency_ret[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ] = vertical_wind_advective_tendency[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        vertical_cfl_ret[horizontal_start:horizontal_end, vertical_start:vertical_end] = (
            vertical_cfl[horizontal_start:horizontal_end, vertical_start:vertical_end]
        )

        return dict(
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels_ret,
            vertical_wind_advective_tendency=vertical_wind_advective_tendency_ret,
            contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels_ret,
            vertical_cfl=vertical_cfl_ret,
        )

    @pytest.fixture(
        params=[
            {"skip_compute_predictor_vertical_advection": value} for value in [True, False]
        ],  # True for benchmarking, False for testing
        ids=lambda param: (
            f"skip_compute_predictor_vertical_advection[{param['skip_compute_predictor_vertical_advection']}]"
        ),
    )
    def input_data(
        self, grid: base.Grid, request: pytest.FixtureRequest
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        contravariant_corrected_w_at_cells_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim
        )
        vertical_wind_advective_tendency = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        horizontal_advection_of_w_at_edges_on_half_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}
        )
        contravariant_correction_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        contravariant_correction_at_cells_on_half_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )

        coeff1_dwdz = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        coeff2_dwdz = data_alloc.random_field(grid, dims.CellDim, dims.KDim)

        e_bln_c_s = data_alloc.random_field(grid, dims.CellDim, dims.C2EDim)
        wgtfac_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim)

        vertical_cfl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        owner_mask = data_alloc.random_mask(grid, dims.CellDim)
        ddqz_z_half = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        area = data_alloc.random_field(grid, dims.CellDim)
        geofac_n2s = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CODim)

        scalfac_exdiff = 10.0
        dtime = 2.0
        cfl_w_limit = 0.65 / dtime

        skip_compute_predictor_vertical_advection = request.param[
            "skip_compute_predictor_vertical_advection"
        ]

        nflatlev = 5  # value is set to reflect the MCH ch1 experiment. Changing this value will change the expected runtime
        end_index_of_damping_layer = 12  # value is set to reflect the MCH ch1 experiment. Changing this value will change the expected runtime

        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.HALO))
        vertical_start = 0
        vertical_end = grid.num_levels

        return dict(
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            vertical_wind_advective_tendency=vertical_wind_advective_tendency,
            contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
            vertical_cfl=vertical_cfl,
            w=w,
            horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            owner_mask=owner_mask,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
            nflatlev=nflatlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )

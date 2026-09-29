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
from icon4py.model.atmosphere.dycore.stencils.compute_advection_in_horizontal_momentum_equation import (
    compute_advection_in_horizontal_momentum,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing import stencil_tests

from .test_interpolate_to_cell_center import interpolate_to_cell_center_numpy
from .test_mo_math_divrot_rot_vertex_ri_dsl import mo_math_divrot_rot_vertex_ri_dsl_numpy


def _compute_advective_normal_wind_tendency_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
    coeff_gradekin: np.ndarray,
    horizontal_kinetic_energy_at_cells_on_model_levels: np.ndarray,
    upward_vorticity_at_vertices: np.ndarray,
    tangential_wind: np.ndarray,
    coriolis_frequency: np.ndarray,
    c_lin_e: np.ndarray,
    contravariant_corrected_w_at_cells_on_model_levels: np.ndarray,
    vn_on_half_levels: np.ndarray,
    ddqz_z_full_e: np.ndarray,
) -> np.ndarray:
    e2c = connectivities[dims.E2CDim]
    horizontal_kinetic_energy_at_cells_on_model_levels_e2c = (
        horizontal_kinetic_energy_at_cells_on_model_levels[e2c]
    )
    coeff_gradekin = np.expand_dims(coeff_gradekin, axis=-1)
    coriolis_frequency = np.expand_dims(coriolis_frequency, axis=-1)
    c_lin_e = np.expand_dims(c_lin_e, axis=-1)

    normal_wind_advective_tendency = -(
        (coeff_gradekin[:, 0] - coeff_gradekin[:, 1])
        * horizontal_kinetic_energy_at_edges_on_model_levels
        + (
            -coeff_gradekin[:, 0] * horizontal_kinetic_energy_at_cells_on_model_levels_e2c[:, 0]
            + coeff_gradekin[:, 1] * horizontal_kinetic_energy_at_cells_on_model_levels_e2c[:, 1]
        )
        + tangential_wind
        * (
            coriolis_frequency
            + 0.5 * np.sum(upward_vorticity_at_vertices[connectivities[dims.E2VDim]], axis=1)
        )
        + np.sum(contravariant_corrected_w_at_cells_on_model_levels[e2c] * c_lin_e, axis=1)
        * (vn_on_half_levels[:, :-1] - vn_on_half_levels[:, 1:])
        / ddqz_z_full_e
    )
    return normal_wind_advective_tendency


def _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_without_levelmask_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    c_lin_e: np.ndarray,
    contravariant_corrected_w_at_cells_on_model_levels: np.ndarray,
    ddqz_z_full_e: np.ndarray,
    area_edge: np.ndarray,
    tangent_orientation: np.ndarray,
    inv_primal_edge_length: np.ndarray,
    upward_vorticity_at_vertices: np.ndarray,
    geofac_grdiv: np.ndarray,
    vn: np.ndarray,
    normal_wind_advective_tendency: np.ndarray,
    cfl_w_limit: ta.wpfloat,
    scalfac_exdiff: ta.wpfloat,
    dtime: ta.wpfloat,
) -> np.ndarray:
    contravariant_corrected_w_at_edges_on_model_levels = np.zeros_like(vn)
    difcoef = np.zeros_like(vn)

    c_lin_e = np.expand_dims(c_lin_e, axis=-1)
    geofac_grdiv = np.expand_dims(geofac_grdiv, axis=-1)
    area_edge = np.expand_dims(area_edge, axis=-1)
    tangent_orientation = np.expand_dims(tangent_orientation, axis=-1)
    inv_primal_edge_length = np.expand_dims(inv_primal_edge_length, axis=-1)

    e2c = connectivities[dims.E2CDim]
    contravariant_corrected_w_at_edges_on_model_levels = np.sum(
        np.where(
            (e2c != -1)[:, :, np.newaxis],
            c_lin_e * contravariant_corrected_w_at_cells_on_model_levels[e2c],
            0,
        ),
        axis=1,
    )

    difcoef = np.where(
        (np.abs(contravariant_corrected_w_at_edges_on_model_levels) > cfl_w_limit * ddqz_z_full_e),
        scalfac_exdiff
        * np.minimum(
            0.85 - cfl_w_limit * dtime,
            np.abs(contravariant_corrected_w_at_edges_on_model_levels) * dtime / ddqz_z_full_e
            - cfl_w_limit * dtime,
        ),
        difcoef,
    )
    e2v = connectivities[dims.E2VDim]
    e2c2eo = connectivities[dims.E2C2EODim]
    normal_wind_advective_tendency = np.where(
        (np.abs(contravariant_corrected_w_at_edges_on_model_levels) > cfl_w_limit * ddqz_z_full_e),
        normal_wind_advective_tendency
        + difcoef
        * area_edge
        * (
            np.sum(
                np.where(
                    (e2c2eo != -1)[:, :, np.newaxis],
                    geofac_grdiv * vn[e2c2eo],
                    0,
                ),
                axis=1,
            )
            + tangent_orientation
            * inv_primal_edge_length
            * (upward_vorticity_at_vertices[e2v][:, 1] - upward_vorticity_at_vertices[e2v][:, 0])
        ),
        normal_wind_advective_tendency,
    )
    return normal_wind_advective_tendency


@pytest.mark.embedded_remap_error
@pytest.mark.continuous_benchmarking
class TestFusedVelocityAdvectionStencilsHMomentum(stencil_tests.StencilTest):
    PROGRAM = compute_advection_in_horizontal_momentum
    OUTPUTS = ("normal_wind_advective_tendency",)
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "end_index_of_damping_layer",
            "vertical_start",
            "vertical_end",
            "apply_extra_diffusion_on_vn",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "end_index_of_damping_layer",
            "vertical_start",
            "vertical_end",
            "apply_extra_diffusion_on_vn",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        normal_wind_advective_tendency: np.ndarray,
        vn: np.ndarray,
        horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
        tangential_wind: np.ndarray,
        coriolis_frequency: np.ndarray,
        contravariant_corrected_w_at_cells_on_model_levels: np.ndarray,
        vn_on_half_levels: np.ndarray,
        e_bln_c_s: np.ndarray,
        geofac_rot: np.ndarray,
        coeff_gradekin: np.ndarray,
        c_lin_e: np.ndarray,
        ddqz_z_full_e: np.ndarray,
        area_edge: np.ndarray,
        tangent_orientation: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        geofac_grdiv: np.ndarray,
        cfl_w_limit: ta.wpfloat,
        scalfac_exdiff: ta.wpfloat,
        dtime: ta.wpfloat,
        apply_extra_diffusion_on_vn: bool,
        end_index_of_damping_layer: int,
        **kwargs: Any,
    ) -> dict:
        normal_wind_advective_tendency_cp = normal_wind_advective_tendency.copy()
        nlev = kwargs["vertical_end"]
        k = np.arange(nlev)

        horizontal_kinetic_energy_at_cells_on_model_levels = interpolate_to_cell_center_numpy(
            connectivities, horizontal_kinetic_energy_at_edges_on_model_levels, e_bln_c_s
        )

        upward_vorticity_at_vertices = mo_math_divrot_rot_vertex_ri_dsl_numpy(
            connectivities, vn, geofac_rot
        )

        normal_wind_advective_tendency = _compute_advective_normal_wind_tendency_numpy(
            connectivities=connectivities,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            coeff_gradekin=coeff_gradekin,
            horizontal_kinetic_energy_at_cells_on_model_levels=horizontal_kinetic_energy_at_cells_on_model_levels,
            upward_vorticity_at_vertices=upward_vorticity_at_vertices,
            tangential_wind=tangential_wind,
            coriolis_frequency=coriolis_frequency,
            c_lin_e=c_lin_e,
            contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
            vn_on_half_levels=vn_on_half_levels,
            ddqz_z_full_e=ddqz_z_full_e,
        )

        condition = (np.maximum(2, end_index_of_damping_layer - 2) <= k) & (k < nlev - 4)

        if apply_extra_diffusion_on_vn:
            normal_wind_advective_tendency_extra_diffu = _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_without_levelmask_numpy(
                connectivities=connectivities,
                c_lin_e=c_lin_e,
                contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
                ddqz_z_full_e=ddqz_z_full_e,
                area_edge=area_edge,
                tangent_orientation=tangent_orientation,
                inv_primal_edge_length=inv_primal_edge_length,
                upward_vorticity_at_vertices=upward_vorticity_at_vertices,
                geofac_grdiv=geofac_grdiv,
                vn=vn,
                normal_wind_advective_tendency=normal_wind_advective_tendency,
                cfl_w_limit=cfl_w_limit,
                scalfac_exdiff=scalfac_exdiff,
                dtime=dtime,
            )

            normal_wind_advective_tendency = np.where(
                condition,
                normal_wind_advective_tendency_extra_diffu,
                normal_wind_advective_tendency,
            )

        # restriction of execution domain
        normal_wind_advective_tendency[0 : kwargs["horizontal_start"], :] = (
            normal_wind_advective_tendency_cp[0 : kwargs["horizontal_start"], :]
        )
        normal_wind_advective_tendency[kwargs["horizontal_end"] :, :] = (
            normal_wind_advective_tendency_cp[kwargs["horizontal_end"] :, :]
        )

        return dict(normal_wind_advective_tendency=normal_wind_advective_tendency)

    @pytest.fixture(
        params=[
            {"apply_extra_diffusion_on_vn": value} for value in [True, False]
        ],  # True for testing, False for benchmarking
        ids=lambda param: f"apply_extra_diffusion_on_vn[{param['apply_extra_diffusion_on_vn']}]",
    )
    def input_data(
        self, request: pytest.FixtureRequest, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        normal_wind_advective_tendency = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        horizontal_kinetic_energy_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        tangential_wind = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        coriolis_frequency = data_alloc.random_field(grid, dims.EdgeDim)
        contravariant_corrected_w_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )
        vn_on_half_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}
        )
        coeff_gradekin = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim)
        e_bln_c_s = data_alloc.random_field(grid, dims.CellDim, dims.C2EDim)
        c_lin_e = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim)
        ddqz_z_full_e = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, low=0.0
        )  # this makes sure that the simplified stencil produces the same result as the numpy version
        area_edge = data_alloc.random_field(grid, dims.EdgeDim)
        tangent_orientation = data_alloc.random_field(grid, dims.EdgeDim)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim)
        geofac_grdiv = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EODim)

        geofac_rot = data_alloc.random_field(grid, dims.VertexDim, dims.V2EDim)
        scalfac_exdiff = 0.6
        dtime = 2.0
        cfl_w_limit = 0.65 / dtime
        apply_extra_diffusion_on_vn = request.param["apply_extra_diffusion_on_vn"]

        end_index_of_damping_layer = 12  # value is set to reflect the MCH ch1 experiment. Changing this value will change the expected runtime
        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.LOCAL))

        return dict(
            normal_wind_advective_tendency=normal_wind_advective_tendency,
            vn=vn,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            tangential_wind=tangential_wind,
            coriolis_frequency=coriolis_frequency,
            contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
            vn_on_half_levels=vn_on_half_levels,
            e_bln_c_s=e_bln_c_s,
            geofac_rot=geofac_rot,
            coeff_gradekin=coeff_gradekin,
            c_lin_e=c_lin_e,
            ddqz_z_full_e=ddqz_z_full_e,
            area_edge=area_edge,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            geofac_grdiv=geofac_grdiv,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            apply_extra_diffusion_on_vn=apply_extra_diffusion_on_vn,
            end_index_of_damping_layer=end_index_of_damping_layer,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

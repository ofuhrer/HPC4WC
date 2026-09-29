# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import gt4py.next as gtx
import pytest


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import model_backends, utils as common_utils
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes as geometry_meta,
    grid_manager as gm,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.states import factory, prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.fixtures.benchmark import (
    geometry_field_source,
    interpolation_field_source,
    metrics_field_source,
)
from icon4py.model.testing.fixtures.datatest import backend_like
from icon4py.model.testing.fixtures.stencil_tests import grid_manager


@pytest.fixture(scope="module")
def solve_nonhydro(
    geometry_field_source: grid_geometry.GridGeometry,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
) -> solve_nh.SolveNonhydro:
    allocator = model_backends.get_allocator(backend_like)
    mesh = geometry_field_source.grid

    config = solve_nh.NonHydrostaticConfig(
        divdamp_order=dycore_states.DivergenceDampingOrder.COMBINED,
        fourth_order_divdamp_factor=0.004,
        max_nudging_coefficient=0.375,
    )

    nonhydro_params = solve_nh.NonHydrostaticParams(config)

    vertical_config = v_grid.VerticalGridConfig(
        mesh.num_levels,
        lowest_layer_thickness=50,
        model_top_height=23500.0,
        stretch_factor=1.0,
        rayleigh_damping_height=1.0,
    )
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, allocator=allocator)

    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )

    cell_geometry = grid_states.CellParams(
        cell_center_lat=geometry_field_source.get(geometry_meta.CELL_LAT),
        cell_center_lon=geometry_field_source.get(geometry_meta.CELL_LON),
        area=geometry_field_source.get(geometry_meta.CELL_AREA),
        mean_cell_area=geometry_field_source.get(geometry_meta.MEAN_CELL_AREA),  # type: ignore[arg-type]
    )
    edge_geometry = grid_states.EdgeParams(
        tangent_orientation=geometry_field_source.get(geometry_meta.TANGENT_ORIENTATION),
        inverse_primal_edge_lengths=geometry_field_source.get(
            f"inverse_of_{geometry_meta.EDGE_LENGTH}"
        ),
        inverse_dual_edge_lengths=geometry_field_source.get(
            f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}"
        ),
        inverse_vertex_vertex_lengths=geometry_field_source.get(
            f"inverse_of_{geometry_meta.VERTEX_VERTEX_LENGTH}"
        ),
        primal_normal_vert_x=geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_U),
        primal_normal_vert_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_V),
        dual_normal_vert_x=geometry_field_source.get(geometry_meta.EDGE_TANGENT_VERTEX_U),
        dual_normal_vert_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_V),
        primal_normal_cell_x=geometry_field_source.get(geometry_meta.EDGE_NORMAL_CELL_U),
        dual_normal_cell_x=geometry_field_source.get(geometry_meta.EDGE_TANGENT_CELL_U),
        primal_normal_cell_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_CELL_V),
        dual_normal_cell_y=geometry_field_source.get(geometry_meta.EDGE_TANGENT_CELL_V),
        edge_areas=geometry_field_source.get(geometry_meta.EDGE_AREA),
        coriolis_frequency=geometry_field_source.get(geometry_meta.CORIOLIS_PARAMETER),
        edge_center_lat=geometry_field_source.get(geometry_meta.EDGE_LAT),
        edge_center_lon=geometry_field_source.get(geometry_meta.EDGE_LON),
        primal_normal_x=geometry_field_source.get(geometry_meta.EDGE_NORMAL_U),
        primal_normal_y=geometry_field_source.get(geometry_meta.EDGE_NORMAL_V),
    )

    interpolation_state = dycore_states.InterpolationState(
        c_lin_e=interpolation_field_source.get(interpolation_attributes.C_LIN_E),
        c_intp=interpolation_field_source.get(interpolation_attributes.CELL_AW_VERTS),
        e_flx_avg=interpolation_field_source.get(interpolation_attributes.E_FLX_AVG),
        geofac_grdiv=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRDIV),
        geofac_rot=interpolation_field_source.get(interpolation_attributes.GEOFAC_ROT),
        pos_on_tplane_e_1=interpolation_field_source.get(
            interpolation_attributes.POS_ON_TPLANE_E_X
        ),
        pos_on_tplane_e_2=interpolation_field_source.get(
            interpolation_attributes.POS_ON_TPLANE_E_Y
        ),
        rbf_vec_coeff_e=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_E),
        e_bln_c_s=interpolation_field_source.get(interpolation_attributes.E_BLN_C_S),
        rbf_coeff_1=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V1),
        rbf_coeff_2=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V2),
        geofac_div=interpolation_field_source.get(interpolation_attributes.GEOFAC_DIV),
        geofac_n2s=interpolation_field_source.get(interpolation_attributes.GEOFAC_N2S),
        geofac_grg_x=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_X),
        geofac_grg_y=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_Y),
        nudgecoeff_e=interpolation_field_source.get(interpolation_attributes.NUDGECOEFFS_E),
    )

    metric_state_nonhydro = dycore_states.MetricStateNonHydro(
        mask_prog_halo_c=metrics_field_source.get(metrics_attributes.MASK_PROG_HALO_C),
        rayleigh_w=metrics_field_source.get(metrics_attributes.RAYLEIGH_W),
        time_extrapolation_parameter_for_exner=metrics_field_source.get(
            metrics_attributes.EXNER_EXFAC
        ),
        reference_exner_at_cells_on_model_levels=metrics_field_source.get(
            metrics_attributes.EXNER_REF_MC
        ),
        wgtfac_c=metrics_field_source.get(metrics_attributes.WGTFAC_C),
        wgtfacq_c=metrics_field_source.get(metrics_attributes.WGTFACQ_C),
        inv_ddqz_z_full=metrics_field_source.get(metrics_attributes.INV_DDQZ_Z_FULL),
        reference_rho_at_cells_on_model_levels=metrics_field_source.get(
            metrics_attributes.RHO_REF_MC
        ),
        reference_theta_at_cells_on_model_levels=metrics_field_source.get(
            metrics_attributes.THETA_REF_MC
        ),
        exner_w_explicit_weight_parameter=metrics_field_source.get(
            metrics_attributes.EXNER_W_EXPLICIT_WEIGHT_PARAMETER
        ),
        ddz_of_reference_exner_at_cells_on_half_levels=metrics_field_source.get(
            metrics_attributes.D_EXNER_DZ_REF_IC
        ),
        ddqz_z_half=metrics_field_source.get(metrics_attributes.DDQZ_Z_HALF),
        reference_theta_at_cells_on_half_levels=metrics_field_source.get(
            metrics_attributes.THETA_REF_IC
        ),
        d2dexdz2_fac1_mc=metrics_field_source.get(metrics_attributes.D2DEXDZ2_FAC1_MC),
        d2dexdz2_fac2_mc=metrics_field_source.get(metrics_attributes.D2DEXDZ2_FAC1_MC),
        reference_rho_at_edges_on_model_levels=metrics_field_source.get(
            metrics_attributes.RHO_REF_ME
        ),
        reference_theta_at_edges_on_model_levels=metrics_field_source.get(
            metrics_attributes.THETA_REF_ME
        ),
        ddxn_z_full=metrics_field_source.get(metrics_attributes.DDXN_Z_FULL),
        zdiff_gradp=metrics_field_source.get(metrics_attributes.ZDIFF_GRADP),
        vertoffset_gradp=metrics_field_source.get(metrics_attributes.VERTOFFSET_GRADP),
        nflat_gradp=metrics_field_source.get_int32(metrics_attributes.NFLAT_GRADP),
        pg_exdist=metrics_field_source.get(metrics_attributes.PG_EXDIST_DSL),
        ddqz_z_full_e=metrics_field_source.get(metrics_attributes.DDQZ_Z_FULL_E),
        ddxt_z_full=metrics_field_source.get(metrics_attributes.DDXT_Z_FULL),
        wgtfac_e=metrics_field_source.get(metrics_attributes.WGTFAC_E),
        wgtfacq_e=metrics_field_source.get(metrics_attributes.WGTFACQ_E),
        exner_w_implicit_weight_parameter=metrics_field_source.get(
            metrics_attributes.EXNER_W_IMPLICIT_WEIGHT_PARAMETER
        ),
        horizontal_mask_for_3d_divdamp=metrics_field_source.get(
            metrics_attributes.HORIZONTAL_MASK_FOR_3D_DIVDAMP
        ),
        scaling_factor_for_3d_divdamp=metrics_field_source.get(
            metrics_attributes.SCALING_FACTOR_FOR_3D_DIVDAMP
        ),
        coeff1_dwdz=metrics_field_source.get(metrics_attributes.COEFF1_DWDZ),
        coeff2_dwdz=metrics_field_source.get(metrics_attributes.COEFF2_DWDZ),
        coeff_gradekin=metrics_field_source.get(metrics_attributes.COEFF_GRADEKIN),
    )

    solve_nonhydro = solve_nh.SolveNonhydro(
        grid=mesh,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_grid,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=geometry_field_source.get("cell_owner_mask"),
        exchange=decomposition.single_node_exchange,
        backend=backend_like,
    )

    return solve_nonhydro


@pytest.mark.parametrize(
    "at_first_substep, at_last_substep", [(True, False), (False, True), (False, False)]
)
@pytest.mark.embedded_remap_error
@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
def test_benchmark_solve_nonhydro(  # noqa: PLR0917 [too-many-positional-arguments]
    grid_manager: gm.GridManager,
    solve_nonhydro: solve_nh.SolveNonhydro,
    at_first_substep: bool,
    at_last_substep: bool,
    backend_like: model_backends.BackendLike,
    benchmark: Any,
) -> None:
    allocator = model_backends.get_allocator(backend_like)
    mesh = grid_manager.grid

    dtime = 10.0 if mesh.limited_area else 90.0

    lprep_adv = True
    ndyn_substeps = 5
    at_initial_timestep = False
    second_order_divdamp_factor = 0.02

    prep_adv = dycore_states.PrepAdvection(
        vn_traj=data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
        mass_flx_me=data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        ),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        ),
    )

    diagnostic_state_nh = dycore_states.DiagnosticStateNonHydro(
        max_vertical_cfl=data_alloc.scalar_like_array(0.0, allocator),
        theta_v_at_cells_on_half_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        ),
        perturbed_exner_at_cells_on_model_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, allocator=allocator
        ),
        rho_at_cells_on_half_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        ),
        exner_tendency_due_to_slow_physics=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, allocator=allocator
        ),
        grf_tend_rho=data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        grf_tend_thv=data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        grf_tend_w=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        ),
        mass_flux_at_edges_on_model_levels=data_alloc.zero_field(
            mesh, dims.EdgeDim, dims.KDim, allocator=allocator
        ),
        normal_wind_tendency_due_to_slow_physics_process=data_alloc.zero_field(
            mesh, dims.EdgeDim, dims.KDim, allocator=allocator
        ),
        grf_tend_vn=data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
            data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            data_alloc.zero_field(
                mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
            ),
            data_alloc.zero_field(
                mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
            ),
        ),
        tangential_wind=data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
        vn_on_half_levels=data_alloc.zero_field(
            mesh, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        ),
        contravariant_correction_at_cells_on_half_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        ),
        rho_iau_increment=data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        normal_wind_iau_increment=data_alloc.zero_field(
            mesh, dims.EdgeDim, dims.KDim, allocator=allocator
        ),
        exner_iau_increment=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, allocator=allocator
        ),
        exner_dynamical_increment=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, allocator=allocator
        ),
    )

    prognostic_state_nnow = prognostics.PrognosticState(
        w=data_alloc.random_field(
            mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        ),
        vn=data_alloc.random_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
        theta_v=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        rho=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        exner=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
    )
    prognostic_state_nnew = prognostics.PrognosticState(
        w=data_alloc.random_field(
            mesh, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        ),
        vn=data_alloc.random_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
        theta_v=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        rho=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        exner=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
    )

    prognostic_states = common_utils.TimeStepPair(prognostic_state_nnow, prognostic_state_nnew)

    solve_nonhydro_timestep_variants = functools.partial(
        solve_nonhydro.time_step,
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        prep_adv=prep_adv,
        second_order_divdamp_factor=second_order_divdamp_factor,
        dtime=dtime,
        ndyn_substeps_var=ndyn_substeps,
        at_initial_timestep=at_initial_timestep,
        lprep_adv=lprep_adv,
    )

    benchmark(
        solve_nonhydro_timestep_variants,
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
    )

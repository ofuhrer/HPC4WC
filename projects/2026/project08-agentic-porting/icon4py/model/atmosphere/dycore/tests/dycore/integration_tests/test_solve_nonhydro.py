# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gt4py.next as gtx
import pytest

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.dycore import dycore_states, dycore_utils, solve_nonhydro as solve_nh
from icon4py.model.atmosphere.dycore.stencils import (
    compute_cell_diagnostics_for_dycore,
    compute_edge_diagnostics_for_dycore_and_update_vn,
    compute_horizontal_velocity_quantities,
    compute_hydrostatic_correction_term,
    vertically_implicit_dycore_solver,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.common.math import smagorinsky
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, test_utils

from .. import utils
from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.MCH_CH_R04B09])
def test_validate_divdamp_fields_against_savepoint_values(
    grid_savepoint: sb.IconGridSavepoint,
    savepoint_nonhydro_init: sb.IconNonHydroInitSavepoint,
    icon_grid: base_grid.Grid,
    backend: gtx_typing.Backend,
) -> None:
    config = solve_nh.NonHydrostaticConfig()
    second_order_divdamp_factor = 0.032
    mean_cell_area = grid_savepoint.mean_cell_area()
    interpolated_fourth_order_divdamp_factor = data_alloc.zero_field(
        icon_grid,
        dims.KDim,
        allocator=backend,
    )
    fourth_order_divdamp_scaling_coeff = data_alloc.zero_field(
        icon_grid,
        dims.KDim,
        allocator=backend,
    )
    reduced_fourth_order_divdamp_coeff_at_nest_boundary = data_alloc.zero_field(
        icon_grid,
        dims.KDim,
        allocator=backend,
    )
    smagorinsky.en_smag_fac_for_zero_nshift.with_backend(backend)(
        grid_savepoint.vct_a(),
        config.fourth_order_divdamp_factor,
        config.fourth_order_divdamp_factor2,
        config.fourth_order_divdamp_factor3,
        config.fourth_order_divdamp_factor4,
        config.fourth_order_divdamp_z,
        config.fourth_order_divdamp_z2,
        config.fourth_order_divdamp_z3,
        config.fourth_order_divdamp_z4,
        interpolated_fourth_order_divdamp_factor,
        offset_provider={},
    )
    dycore_utils._calculate_fourth_order_divdamp_scaling_coeff.with_backend(backend)(
        interpolated_fourth_order_divdamp_factor=interpolated_fourth_order_divdamp_factor,
        divdamp_order=config.divdamp_order,
        mean_cell_area=mean_cell_area,
        second_order_divdamp_factor=second_order_divdamp_factor,
        out=fourth_order_divdamp_scaling_coeff,
        offset_provider={},
    )
    dycore_utils._calculate_reduced_fourth_order_divdamp_coeff_at_nest_boundary.with_backend(
        backend
    )(
        fourth_order_divdamp_scaling_coeff,
        config.max_nudging_coefficient,
        constants.DBL_EPS,
        out=reduced_fourth_order_divdamp_coeff_at_nest_boundary,
        offset_provider={},
    )

    assert test_utils.dallclose(
        fourth_order_divdamp_scaling_coeff.asnumpy(),
        savepoint_nonhydro_init.scal_divdamp().asnumpy(),
    )
    assert test_utils.dallclose(
        reduced_fourth_order_divdamp_coeff_at_nest_boundary.asnumpy(),
        savepoint_nonhydro_init.bdy_divdamp().asnumpy(),
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.MCH_CH_R04B09])
@pytest.mark.parametrize(
    "istep_init, step_date_init, substep_init, at_initial_timestep",
    [
        (1, "2021-06-20T12:00:10.000", 1, True),
        (2, "2021-06-20T12:00:10.000", 1, True),
        (1, "2021-06-20T12:00:10.000", 2, True),
        (2, "2021-06-20T12:00:10.000", 2, True),
        (1, "2021-06-20T12:00:20.000", 1, False),
        (2, "2021-06-20T12:00:20.000", 1, False),
        (1, "2021-06-20T12:00:20.000", 2, False),
        (2, "2021-06-20T12:00:20.000", 2, False),
    ],
)
def test_time_step_flags(
    substep_init: int,
    at_initial_timestep: bool,
    savepoint_nonhydro_init: sb.IconNonHydroInitSavepoint,
) -> None:
    sp = savepoint_nonhydro_init

    recompute = sp.get_metadata("recompute").get("recompute")
    clean_mflx = sp.get_metadata("clean_mflx").get("clean_mflx")
    linit = sp.get_metadata("linit").get("linit")

    assert recompute == (substep_init == 1)
    assert clean_mflx == (substep_init == 1)
    assert linit == (at_initial_timestep and (substep_init == 1))


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("at_initial_timestep", [True])
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            definitions.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_nonhydro_predictor_step(  # noqa: PLR0917 [too-many-positional-arguments]
    istep_init,
    istep_exit,
    substep_init,
    step_date_init,
    step_date_exit,
    icon_grid,
    savepoint_nonhydro_init,
    is_iau_active,
    iau_wgt_dyn,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    experiment,
    at_initial_timestep,
    caplog,
    backend,
):
    caplog.set_level(logging.WARN)
    config = experiment.config.nonhydrostatic
    sp = savepoint_nonhydro_init
    sp_exit = savepoint_nonhydro_exit
    nonhydro_params = solve_nh.NonHydrostaticParams(config)
    vertical_config = experiment.config.vertical_grid
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)
    dtime = sp.get_metadata("dtime").get("dtime")

    diagnostic_state_nh = utils.construct_diagnostics(sp, icon_grid, backend)

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, grid_savepoint)

    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = solve_nh.SolveNonhydro(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        exchange=decomp_defs.single_node_exchange,
        backend=backend,
    )
    nlev = icon_grid.num_levels
    at_first_substep = substep_init == 1

    prognostic_states = utils.create_prognostic_states(sp)

    if not (at_initial_timestep and at_first_substep):
        diagnostic_state_nh.vertical_wind_advective_tendency.swap()
    if not at_first_substep:
        diagnostic_state_nh.normal_wind_advective_tendency.swap()

    solve_nonhydro.run_predictor_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        z_fields=solve_nonhydro.intermediate_fields,
        dtime=dtime,
        at_initial_timestep=at_initial_timestep,
        at_first_substep=at_first_substep,
        is_iau_active=is_iau_active,
        iau_wgt_dyn=iau_wgt_dyn,
    )

    cell_domain = h_grid.domain(dims.CellDim)
    edge_domain = h_grid.domain(dims.EdgeDim)

    cell_start_lateral_boundary_level_2 = icon_grid.start_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )

    edge_start_lateral_boundary_level_5 = icon_grid.start_index(
        edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
    )

    edge_start_lateral_boundary_level_7 = icon_grid.start_index(
        edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
    )
    edge_start_nudging_level_2 = icon_grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))

    # stencils 2, 3
    assert test_utils.dallclose(
        diagnostic_state_nh.perturbed_exner_at_cells_on_model_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, :
        ],
        sp_exit.exner_pr().asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    assert test_utils.dallclose(
        solve_nonhydro.temporal_extrapolation_of_perturbed_exner.asnumpy()[
            cell_start_lateral_boundary_level_2:, :
        ],
        sp_exit.z_exner_ex_pr().asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )

    # stencils 4,5
    assert test_utils.dallclose(
        solve_nonhydro.exner_at_cells_on_half_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, nlev - 1
        ],
        sp_exit.z_exner_ic().asnumpy()[cell_start_lateral_boundary_level_2:, nlev - 1],
    )
    nflatlev = vertical_params.nflatlev
    assert test_utils.dallclose(
        solve_nonhydro.exner_at_cells_on_half_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, nflatlev : nlev - 1
        ],
        sp_exit.z_exner_ic().asnumpy()[cell_start_lateral_boundary_level_2:, nflatlev : nlev - 1],
        rtol=1.0e-9,
    )
    # stencil 6
    assert test_utils.dallclose(
        solve_nonhydro.ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, nflatlev:
        ],
        sp_exit.z_dexner_dz_c(0).asnumpy()[cell_start_lateral_boundary_level_2:, nflatlev:],
        atol=5e-18,
    )

    # stencils 7,8,9
    assert test_utils.dallclose(
        diagnostic_state_nh.rho_at_cells_on_half_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, :
        ],
        sp_exit.rho_ic().asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    assert test_utils.dallclose(
        solve_nonhydro.nonhydro_buoy_at_cells_on_half_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, 1:
        ],
        sp_exit.z_th_ddz_exner_c().asnumpy()[cell_start_lateral_boundary_level_2:, 1:],
        rtol=2.0e-12,
    )

    # stencils 7,8,9, 11
    assert test_utils.dallclose(
        solve_nonhydro.perturbed_theta_v_at_cells_on_half_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, :
        ],
        sp_exit.z_theta_v_pr_ic().asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    assert test_utils.dallclose(
        diagnostic_state_nh.theta_v_at_cells_on_half_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, :
        ],
        sp_exit.theta_v_ic().asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    # stencils 7,8,9, 13
    assert test_utils.dallclose(
        solve_nonhydro.perturbed_rho_at_cells_on_model_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, :
        ],
        sp_exit.z_rth_pr(0).asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    assert test_utils.dallclose(
        solve_nonhydro.perturbed_theta_v_at_cells_on_model_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, :
        ],
        sp_exit.z_rth_pr(1).asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )

    # stencils 12
    nflat_gradp = grid_savepoint.nflat_gradp()
    assert test_utils.dallclose(
        solve_nonhydro.d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, nflat_gradp:
        ],
        sp_exit.z_dexner_dz_c(1).asnumpy()[cell_start_lateral_boundary_level_2:, nflat_gradp:],
        atol=1e-22,
    )

    # compute_horizontal_advection_of_rho_and_theta
    assert test_utils.dallclose(
        solve_nonhydro.intermediate_fields.rho_at_edges_on_model_levels.asnumpy()[
            edge_start_lateral_boundary_level_7:, :
        ],
        sp_exit.z_rho_e().asnumpy()[edge_start_lateral_boundary_level_7:, :],
    )
    assert test_utils.dallclose(
        solve_nonhydro.intermediate_fields.theta_v_at_edges_on_model_levels.asnumpy()[
            edge_start_lateral_boundary_level_7:, :
        ],
        sp_exit.z_theta_v_e().asnumpy()[edge_start_lateral_boundary_level_7:, :],
    )

    # stencils 18,19, 20, 22
    assert test_utils.dallclose(
        solve_nonhydro.intermediate_fields.horizontal_pressure_gradient.asnumpy()[
            edge_start_nudging_level_2:, :
        ],
        sp_exit.z_gradh_exner().asnumpy()[edge_start_nudging_level_2:, :],
        atol=1e-20,
    )
    prognostic_state_nnew = prognostic_states.next
    vn_new_reference = sp_exit.vn_new().asnumpy()

    # stencils 24
    assert test_utils.dallclose(
        prognostic_state_nnew.vn.asnumpy()[edge_start_nudging_level_2:, :],
        vn_new_reference[edge_start_nudging_level_2:, :],
        atol=6e-15,
    )
    # stencil 29
    assert test_utils.dallclose(
        prognostic_state_nnew.vn.asnumpy()[:edge_start_nudging_level_2, :],
        vn_new_reference[:edge_start_nudging_level_2, :],
    )

    # stencil 30
    assert test_utils.dallclose(
        solve_nonhydro.z_vn_avg.asnumpy()[edge_start_lateral_boundary_level_5:, :],
        sp_exit.z_vn_avg().asnumpy()[edge_start_lateral_boundary_level_5:, :],
        atol=5e-14,
    )
    # stencil 30
    assert test_utils.dallclose(
        solve_nonhydro.intermediate_fields.horizontal_gradient_of_normal_wind_divergence.asnumpy()[
            edge_start_lateral_boundary_level_5:, :
        ],
        sp_exit.z_graddiv_vn().asnumpy()[edge_start_lateral_boundary_level_5:, :],
        atol=5e-20,
    )
    # stencil 30
    assert test_utils.dallclose(
        diagnostic_state_nh.tangential_wind.asnumpy(),
        sp_exit.vt().asnumpy(),
        atol=5e-14,
    )

    # stencil 32
    assert test_utils.dallclose(
        diagnostic_state_nh.mass_flux_at_edges_on_model_levels.asnumpy(),
        sp_exit.mass_fl_e().asnumpy(),
        atol=4e-12,
    )
    # stencil 32
    assert test_utils.dallclose(
        solve_nonhydro.theta_v_flux_at_edges_on_model_levels.asnumpy()[
            edge_start_lateral_boundary_level_5:, :
        ],
        sp_exit.z_theta_v_fl_e().asnumpy()[edge_start_lateral_boundary_level_5:, :],
        atol=1e-9,
    )

    # stencil 35,36, 37,38
    assert test_utils.dallclose(
        diagnostic_state_nh.vn_on_half_levels.asnumpy()[edge_start_lateral_boundary_level_5:, :],
        sp_exit.vn_ie().asnumpy()[edge_start_lateral_boundary_level_5:, :],
        atol=2e-14,
    )

    # stencil 35,36, 37,38
    assert test_utils.dallclose(
        solve_nonhydro.intermediate_fields.tangential_wind_on_half_levels.asnumpy()[
            edge_start_lateral_boundary_level_5:, :
        ],
        sp_exit.z_vt_ie().asnumpy()[edge_start_lateral_boundary_level_5:, :],
        atol=2e-14,
    )
    # stencil 35,36
    assert test_utils.dallclose(
        solve_nonhydro.intermediate_fields.horizontal_kinetic_energy_at_edges_on_model_levels.asnumpy()[
            edge_start_lateral_boundary_level_5:, :
        ],
        sp_exit.z_kin_hor_e().asnumpy()[edge_start_lateral_boundary_level_5:, :],
        atol=1e-20,
    )
    # stencil 35
    assert test_utils.dallclose(
        solve_nonhydro._contravariant_correction_at_edges_on_model_levels.asnumpy()[
            edge_start_lateral_boundary_level_5:, nflatlev:
        ],
        sp_exit.z_w_concorr_me().asnumpy()[edge_start_lateral_boundary_level_5:, nflatlev:],
        atol=1e-15,
    )

    # stencils 39,40
    assert test_utils.dallclose(
        diagnostic_state_nh.contravariant_correction_at_cells_on_half_levels.asnumpy(),
        sp_exit.w_concorr_c().asnumpy(),
        atol=1e-15,
    )

    # end
    assert test_utils.dallclose(prognostic_state_nnew.rho.asnumpy(), sp_exit.rho_new().asnumpy())
    assert test_utils.dallclose(
        prognostic_state_nnew.w.asnumpy(), sp_exit.w_new().asnumpy(), atol=7e-14
    )

    assert test_utils.dallclose(
        prognostic_state_nnew.exner.asnumpy(), sp_exit.exner_new().asnumpy()
    )
    assert test_utils.dallclose(
        prognostic_state_nnew.theta_v.asnumpy(), sp_exit.theta_v_new().asnumpy()
    )


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep_init, substep_init, istep_exit, substep_exit, at_initial_timestep", [(2, 1, 2, 1, True)]
)
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            definitions.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_nonhydro_corrector_step(  # noqa: PLR0917 [too-many-positional-arguments]
    istep_init,
    substep_init,
    istep_exit,
    substep_exit,
    at_initial_timestep,
    step_date_init,
    step_date_exit,
    icon_grid,
    savepoint_nonhydro_init,
    is_iau_active,
    iau_wgt_dyn,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    experiment,
    caplog,
    backend,
):
    caplog.set_level(logging.WARN)
    config = experiment.config.nonhydrostatic
    init_savepoint = savepoint_nonhydro_init
    nonhydro_params = solve_nh.NonHydrostaticParams(config)
    vertical_config = experiment.config.vertical_grid
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)
    dtime = init_savepoint.get_metadata("dtime").get("dtime")
    lprep_adv = init_savepoint.get_metadata("prep_adv").get("prep_adv")
    prep_adv = dycore_states.PrepAdvection(
        vn_traj=init_savepoint.vn_traj(),
        mass_flx_me=init_savepoint.mass_flx_me(),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=init_savepoint.mass_flx_ic(),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, allocator=backend
        ),
    )

    diagnostic_state_nh = utils.construct_diagnostics(init_savepoint, icon_grid, backend)

    z_fields = solve_nh.IntermediateFields(
        horizontal_pressure_gradient=init_savepoint.z_gradh_exner(),
        rho_at_edges_on_model_levels=init_savepoint.z_rho_e(),
        theta_v_at_edges_on_model_levels=init_savepoint.z_theta_v_e(),
        horizontal_gradient_of_normal_wind_divergence=init_savepoint.z_graddiv_vn(),
        dwdz_at_cells_on_model_levels=init_savepoint.z_dwdz_dd(),
        horizontal_kinetic_energy_at_edges_on_model_levels=init_savepoint.z_kin_hor_e(),
        tangential_wind_on_half_levels=init_savepoint.z_vt_ie(),
    )

    second_order_divdamp_factor = init_savepoint.divdamp_fac_o2()

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, grid_savepoint)

    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = solve_nh.SolveNonhydro(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        exchange=decomp_defs.single_node_exchange,
        backend=backend,
    )
    at_first_substep = substep_init == 1
    at_last_substep = substep_init == experiment.config.diffusion.ndyn_substeps

    prognostic_states = utils.create_prognostic_states(init_savepoint)

    if not (at_initial_timestep and at_first_substep):
        diagnostic_state_nh.vertical_wind_advective_tendency.swap()
    if not at_first_substep:
        diagnostic_state_nh.normal_wind_advective_tendency.swap()

    solve_nonhydro.run_corrector_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        z_fields=z_fields,
        prep_adv=prep_adv,
        second_order_divdamp_factor=second_order_divdamp_factor,
        dtime=dtime,
        ndyn_substeps_var=experiment.config.diffusion.ndyn_substeps,
        lprep_adv=lprep_adv,
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
        is_iau_active=is_iau_active,
        iau_wgt_dyn=iau_wgt_dyn,
    )

    # stencil 10
    assert test_utils.dallclose(
        diagnostic_state_nh.rho_at_cells_on_half_levels.asnumpy(),
        savepoint_nonhydro_exit.rho_ic().asnumpy(),
    )
    # stencil 10
    assert test_utils.dallclose(
        diagnostic_state_nh.theta_v_at_cells_on_half_levels.asnumpy(),
        savepoint_nonhydro_exit.theta_v_ic().asnumpy(),
        atol=1.0e-12,
    )

    # stencil 23,26, 27, 4th_order_divdamp
    assert test_utils.dallclose(
        prognostic_states.next.vn.asnumpy(),
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        rtol=1e-9,  # TODO(halungge): was 1e-10 for local experiment only
    )

    assert test_utils.dallclose(
        prognostic_states.next.exner.asnumpy(),
        savepoint_nonhydro_exit.exner_new().asnumpy(),
    )

    assert test_utils.dallclose(
        prognostic_states.next.rho.asnumpy(),
        savepoint_nonhydro_exit.rho_new().asnumpy(),
    )

    assert test_utils.dallclose(
        prognostic_states.next.w.asnumpy(),
        savepoint_nonhydro_exit.w_new().asnumpy(),
        atol=8e-14,
    )

    assert test_utils.dallclose(
        prognostic_states.next.theta_v.asnumpy(),
        savepoint_nonhydro_exit.theta_v_new().asnumpy(),
    )
    # stencil 31
    assert test_utils.dallclose(
        solve_nonhydro.z_vn_avg.asnumpy()[solve_nonhydro._start_edge_lateral_boundary_level_5 :, :],
        savepoint_nonhydro_exit.z_vn_avg().asnumpy()[
            solve_nonhydro._start_edge_lateral_boundary_level_5 :, :
        ],
        rtol=5e-7,
    )

    # stencil 32
    assert test_utils.dallclose(
        diagnostic_state_nh.mass_flux_at_edges_on_model_levels.asnumpy(),
        savepoint_nonhydro_exit.mass_fl_e().asnumpy(),
        rtol=5e-7,  # TODO(halungge): was rtol=1e-10 for local experiment only
    )

    # stencil 33, 34
    assert test_utils.dallclose(
        prep_adv.mass_flx_me.asnumpy(),
        savepoint_nonhydro_exit.mass_flx_me().asnumpy(),
        rtol=5e-7,  # TODO(halungge): was rtol=1e-10 for local experiment only
    )
    # stencil 33, 34
    assert test_utils.dallclose(
        prep_adv.vn_traj.asnumpy(),
        savepoint_nonhydro_exit.vn_traj().asnumpy(),
        rtol=5e-7,  # TODO(halungge): was rtol=1e-10 for local experiment only
    )
    # stencil 60 only relevant for last substep
    assert test_utils.dallclose(
        diagnostic_state_nh.exner_dynamical_increment.asnumpy(),
        savepoint_nonhydro_exit.exner_dyn_incr().asnumpy(),
        atol=1e-14,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep_init, substep_init, istep_exit, substep_exit, at_initial_timestep", [(1, 1, 2, 1, True)]
)
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            definitions.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_run_solve_nonhydro_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    istep_init,
    substep_init,
    istep_exit,
    substep_exit,
    at_initial_timestep,
    step_date_init,
    step_date_exit,
    experiment,
    icon_grid,
    savepoint_nonhydro_init,
    is_iau_active,
    iau_wgt_dyn,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_step_final,
    caplog,
    backend,
):
    caplog.set_level(logging.WARN)
    config = experiment.config.nonhydrostatic

    sp = savepoint_nonhydro_init
    sp_step_exit = savepoint_nonhydro_step_final
    nonhydro_params = solve_nh.NonHydrostaticParams(config)
    vertical_config = experiment.config.vertical_grid
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)
    dtime = sp.get_metadata("dtime").get("dtime")
    lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")
    prep_adv = dycore_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=sp.mass_flx_ic(),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, allocator=backend
        ),
    )

    diagnostic_state_nh = utils.construct_diagnostics(sp, icon_grid, backend)

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, grid_savepoint)

    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = solve_nh.SolveNonhydro(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        exchange=decomp_defs.single_node_exchange,
        backend=backend,
    )

    prognostic_states = utils.create_prognostic_states(sp)

    second_order_divdamp_factor = sp.divdamp_fac_o2()
    solve_nonhydro.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        prep_adv=prep_adv,
        second_order_divdamp_factor=second_order_divdamp_factor,
        dtime=dtime,
        ndyn_substeps_var=experiment.config.diffusion.ndyn_substeps,
        at_initial_timestep=at_initial_timestep,
        lprep_adv=lprep_adv,
        at_first_substep=substep_init == 1,
        at_last_substep=substep_init == experiment.config.diffusion.ndyn_substeps,
        is_iau_active=is_iau_active,
        iau_wgt_dyn=iau_wgt_dyn,
    )
    prognostic_state_nnew = prognostic_states.next
    assert test_utils.dallclose(
        prognostic_state_nnew.theta_v.asnumpy(),
        sp_step_exit.theta_v_new().asnumpy(),
    )

    assert test_utils.dallclose(
        prognostic_state_nnew.exner.asnumpy(), sp_step_exit.exner_new().asnumpy()
    )

    assert test_utils.dallclose(
        prognostic_state_nnew.vn.asnumpy(),
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        rtol=1e-12,
        atol=1e-13,
    )

    assert test_utils.dallclose(
        prognostic_state_nnew.rho.asnumpy(), savepoint_nonhydro_exit.rho_new().asnumpy()
    )

    assert test_utils.dallclose(
        prognostic_state_nnew.w.asnumpy(),
        savepoint_nonhydro_exit.w_new().asnumpy(),
        atol=8e-14,
    )

    assert test_utils.dallclose(
        diagnostic_state_nh.exner_dynamical_increment.asnumpy(),
        savepoint_nonhydro_exit.exner_dyn_incr().asnumpy(),
        atol=1e-14,
    )


# why is this not run for APE?
@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.MCH_CH_R04B09])
@pytest.mark.parametrize(
    "istep_init, substep_init, step_date_init, istep_exit, substep_exit, step_date_exit, at_initial_timestep",
    [
        (1, 1, "2021-06-20T12:00:10.000", 2, 2, "2021-06-20T12:00:10.000", True),
        (1, 1, "2021-06-20T12:00:20.000", 2, 2, "2021-06-20T12:00:20.000", False),
    ],
)
def test_run_solve_nonhydro_multi_step(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment,
    istep_init,
    substep_init,
    step_date_init,
    istep_exit,
    substep_exit,
    step_date_exit,
    at_initial_timestep,
    icon_grid,
    savepoint_nonhydro_init,
    is_iau_active,
    iau_wgt_dyn,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_step_final,
    backend,
):
    config = experiment.config.nonhydrostatic
    sp = savepoint_nonhydro_init
    sp_step_exit = savepoint_nonhydro_step_final
    nonhydro_params = solve_nh.NonHydrostaticParams(config)
    vertical_config = experiment.config.vertical_grid
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)
    dtime = sp.get_metadata("dtime").get("dtime")
    lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")
    prep_adv = dycore_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=sp.mass_flx_ic(),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, allocator=backend
        ),
    )

    linit = sp.get_metadata("linit").get("linit")

    diagnostic_state_nh = utils.construct_diagnostics(
        sp, icon_grid, backend, swap_vertical_wind_advective_tendency=not linit
    )
    prognostic_states = utils.create_prognostic_states(sp)

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, grid_savepoint)

    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = solve_nh.SolveNonhydro(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        exchange=decomp_defs.single_node_exchange,
        backend=backend,
    )

    for i_substep in range(experiment.config.diffusion.ndyn_substeps):
        at_first_substep = i_substep == 0
        at_last_substep = i_substep == (experiment.config.diffusion.ndyn_substeps - 1)

        if not (at_initial_timestep and at_first_substep):
            diagnostic_state_nh.vertical_wind_advective_tendency.swap()
        if not at_first_substep:
            diagnostic_state_nh.normal_wind_advective_tendency.swap()

        solve_nonhydro.time_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_states=prognostic_states,
            prep_adv=prep_adv,
            second_order_divdamp_factor=sp.divdamp_fac_o2(),
            dtime=dtime,
            ndyn_substeps_var=experiment.config.diffusion.ndyn_substeps,
            at_initial_timestep=at_initial_timestep,
            lprep_adv=lprep_adv,
            at_first_substep=at_first_substep,
            at_last_substep=at_last_substep,
            is_iau_active=is_iau_active,
            iau_wgt_dyn=iau_wgt_dyn,
        )

        if not at_last_substep:
            prognostic_states.swap()

    cell_start_lb_plus2 = icon_grid.start_index(
        h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )
    edge_start_lb_plus4 = icon_grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
    )

    assert test_utils.dallclose(
        diagnostic_state_nh.rho_at_cells_on_half_levels.asnumpy()[cell_start_lb_plus2:, :],
        savepoint_nonhydro_exit.rho_ic().asnumpy()[cell_start_lb_plus2:, :],
    )

    assert test_utils.dallclose(
        diagnostic_state_nh.theta_v_at_cells_on_half_levels.asnumpy()[cell_start_lb_plus2:, :],
        savepoint_nonhydro_exit.theta_v_ic().asnumpy()[cell_start_lb_plus2:, :],
    )

    assert test_utils.dallclose(
        diagnostic_state_nh.mass_flux_at_edges_on_model_levels.asnumpy()[edge_start_lb_plus4:, :],
        savepoint_nonhydro_exit.mass_fl_e().asnumpy()[edge_start_lb_plus4:, :],
        atol=5e-7,
    )

    assert test_utils.dallclose(
        prep_adv.mass_flx_me.asnumpy(),
        savepoint_nonhydro_exit.mass_flx_me().asnumpy(),
        atol=5e-7,
    )

    assert test_utils.dallclose(
        prep_adv.vn_traj.asnumpy(),
        savepoint_nonhydro_exit.vn_traj().asnumpy(),
        atol=1e-12,
    )

    assert test_utils.dallclose(
        prognostic_states.next.theta_v.asnumpy(),
        sp_step_exit.theta_v_new().asnumpy(),
    )

    assert test_utils.dallclose(
        prognostic_states.next.rho.asnumpy(),
        savepoint_nonhydro_exit.rho_new().asnumpy(),
    )

    assert test_utils.dallclose(
        prognostic_states.next.exner.asnumpy(),
        sp_step_exit.exner_new().asnumpy(),
    )

    assert test_utils.dallclose(
        prognostic_states.next.w.asnumpy(),
        savepoint_nonhydro_exit.w_new().asnumpy(),
        atol=1e-13,
    )

    assert test_utils.dallclose(
        prognostic_states.next.vn.asnumpy(),
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        atol=5e-13,
    )
    assert test_utils.dallclose(
        diagnostic_state_nh.exner_dynamical_increment.asnumpy(),
        savepoint_nonhydro_exit.exner_dyn_incr().asnumpy(),
        atol=1e-14,
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.MCH_CH_R04B09])
def test_non_hydrostatic_params(savepoint_nonhydro_init):
    config = solve_nh.NonHydrostaticConfig()
    params = solve_nh.NonHydrostaticParams(config)

    assert params.advection_implicit_weight_parameter == savepoint_nonhydro_init.wgt_nnew_vel()
    assert params.advection_explicit_weight_parameter == savepoint_nonhydro_init.wgt_nnow_vel()
    assert params.rhotheta_implicit_weight_parameter == savepoint_nonhydro_init.wgt_nnew_rth()
    assert params.rhotheta_explicit_weight_parameter == savepoint_nonhydro_init.wgt_nnow_rth()


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("at_initial_timestep", [True])
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            definitions.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_compute_perturbed_quantities_and_interpolation(  # noqa: PLR0917 [too-many-positional-arguments]
    at_initial_timestep,
    experiment,
    step_date_init,
    step_date_exit,
    icon_grid,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    substep_init,
    substep_exit,
    savepoint_nonhydro_init,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init,
    savepoint_nonhydro_exit,
    backend,
):
    sp_init = savepoint_nonhydro_init
    sp_ref = savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init
    sp_exit = savepoint_nonhydro_exit
    vertical_config = experiment.config.vertical_grid
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)

    current_rho = sp_init.rho_now()
    current_theta_v = sp_init.theta_v_now()
    perturbed_exner_at_cells_on_model_levels = sp_init.exner_pr()
    rho_at_cells_on_half_levels = sp_init.rho_ic()
    current_exner = sp_init.exner_now()
    theta_v_at_cells_on_half_levels = sp_init.theta_v_ic()

    # local fields
    perturbed_rho_at_cells_on_model_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, allocator=backend
    )
    perturbed_theta_v_at_cells_on_model_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, allocator=backend
    )
    perturbed_theta_v_at_cells_on_half_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
    )
    nonhydro_buoy_at_cells_on_half_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, allocator=backend
    )
    exner_at_cells_on_half_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
    )
    temporal_extrapolation_of_perturbed_exner = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
    )
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, allocator=backend
    )
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, allocator=backend
    )

    config = experiment.config.nonhydrostatic
    igradp_method = config.igradp_method

    nflatlev = vertical_params.nflatlev
    nflat_gradp = grid_savepoint.nflat_gradp()

    cell_domain = h_grid.domain(dims.CellDim)
    start_cell_lateral_boundary = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY))
    start_cell_lateral_boundary_level_3 = icon_grid.start_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )
    start_cell_halo_level_2 = icon_grid.start_index(cell_domain(h_grid.Zone.HALO_LEVEL_2))
    end_cell_halo = icon_grid.end_index(cell_domain(h_grid.Zone.HALO))
    end_cell_halo_level_2 = icon_grid.end_index(cell_domain(h_grid.Zone.HALO_LEVEL_2))

    reference_rho_at_cells_on_model_levels = metrics_savepoint.rho_ref_mc()
    reference_theta_at_cells_on_model_levels = metrics_savepoint.theta_ref_mc()
    reference_theta_at_cells_on_half_levels = metrics_savepoint.theta_ref_ic()
    d2dexdz2_fac1_mc = metrics_savepoint.d2dexdz2_fac1_mc()
    d2dexdz2_fac2_mc = metrics_savepoint.d2dexdz2_fac2_mc()
    wgtfacq_c = metrics_savepoint.wgtfacq_c()
    wgtfac_c = metrics_savepoint.wgtfac_c()
    exner_w_explicit_weight_parameter = metrics_savepoint.vwind_expl_wgt()
    ddz_of_reference_exner_at_cells_on_half_levels = metrics_savepoint.d_exner_dz_ref_ic()
    ddqz_z_half = metrics_savepoint.ddqz_z_half()
    time_extrapolation_parameter_for_exner = metrics_savepoint.exner_exfac()
    reference_exner_at_cells_on_model_levels = metrics_savepoint.exner_ref_mc()
    inv_ddqz_z_full = metrics_savepoint.inv_ddqz_z_full()

    z_rth_pr_1_ref = sp_ref.z_rth_pr(0)
    z_rth_pr_2_ref = sp_ref.z_rth_pr(1)
    z_exner_ex_pr_ref = sp_ref.z_exner_ex_pr()
    exner_pr_ref = sp_exit.exner_pr()
    rho_ic_ref = sp_exit.rho_ic()
    z_exner_ic_ref = sp_exit.z_exner_ic()
    z_theta_v_pr_ic_ref = sp_exit.z_theta_v_pr_ic()
    theta_v_ic_ref = sp_ref.theta_v_ic()
    z_dexner_dz_c_1_ref = sp_ref.z_dexner_dz_c(0)
    z_dexner_dz_c_2_ref = sp_ref.z_dexner_dz_c(1)

    compute_cell_diagnostics_for_dycore.compute_perturbed_quantities_and_interpolation.with_backend(
        backend
    )(
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
        surface_level=icon_grid.num_levels + 1,
        offset_provider={},
    )
    lb = start_cell_lateral_boundary_level_3

    assert test_utils.dallclose(
        perturbed_rho_at_cells_on_model_levels.asnumpy(), z_rth_pr_1_ref.asnumpy()
    )
    assert test_utils.dallclose(
        perturbed_theta_v_at_cells_on_model_levels.asnumpy(), z_rth_pr_2_ref.asnumpy()
    )
    # `z_exner_ex_pr` is only computed in a subset of the whole domain, reference may contain garbage outside this range
    assert test_utils.dallclose(
        temporal_extrapolation_of_perturbed_exner.asnumpy()[
            start_cell_lateral_boundary_level_3:end_cell_halo, :
        ],
        z_exner_ex_pr_ref.asnumpy()[start_cell_lateral_boundary_level_3:end_cell_halo, :],
    )
    assert test_utils.dallclose(
        perturbed_exner_at_cells_on_model_levels.asnumpy(), exner_pr_ref.asnumpy()
    )
    assert test_utils.dallclose(rho_at_cells_on_half_levels.asnumpy(), rho_ic_ref.asnumpy())

    # `exner_at_cells_on_half_levels` is only computed in a subset of the whole domain, reference may contain garbage outside this range
    assert test_utils.dallclose(
        exner_at_cells_on_half_levels.asnumpy()[
            start_cell_lateral_boundary_level_3:end_cell_halo, nflatlev:
        ],
        z_exner_ic_ref.asnumpy()[start_cell_lateral_boundary_level_3:end_cell_halo, nflatlev:],
        rtol=1e-11,
    )

    assert test_utils.dallclose(
        perturbed_theta_v_at_cells_on_half_levels.asnumpy()[lb:, :],
        z_theta_v_pr_ic_ref.asnumpy()[lb:, :],
    )
    assert test_utils.dallclose(
        theta_v_at_cells_on_half_levels.asnumpy()[lb:, :], theta_v_ic_ref.asnumpy()[lb:, :]
    )

    assert test_utils.dallclose(
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels.asnumpy()[lb:, nflatlev:],
        z_dexner_dz_c_1_ref.asnumpy()[lb:, nflatlev:],
        rtol=5e-9,
    )
    assert test_utils.dallclose(
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels.asnumpy()[
            lb:, nflat_gradp:
        ],
        z_dexner_dz_c_2_ref.asnumpy()[lb:, nflat_gradp:],
        rtol=5e-9,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("at_initial_timestep, istep_init, istep_exit", [(True, 2, 2)])
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            definitions.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_compute_interpolation_and_nonhydro_buoy(  # noqa: PLR0917 [too-many-positional-arguments]
    at_initial_timestep,
    istep_init,
    istep_exit,
    experiment,
    step_date_init,
    step_date_exit,
    icon_grid,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    substep_init,
    substep_exit,
    savepoint_nonhydro_init,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init,
    savepoint_nonhydro_exit,
    backend,
):
    sp_init = savepoint_nonhydro_init
    sp_ref = savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init
    sp_exit = savepoint_nonhydro_exit

    dtime = sp_init.get_metadata("dtime").get("dtime")
    current_rho = sp_init.rho_now()
    next_rho = sp_init.rho_new()

    w = sp_init.w_new()
    contravariant_correction_at_cells_on_half_levels = sp_init.w_concorr_c()
    current_theta_v = sp_init.theta_v_now()
    next_theta_v = sp_init.theta_v_new()
    perturbed_exner_at_cells_on_model_levels = sp_init.exner_pr()
    rho_at_cells_on_half_levels = sp_init.rho_ic()
    theta_v_at_cells_on_half_levels = sp_init.theta_v_ic()
    rhotheta_explicit_weight_parameter = sp_init.wgt_nnow_rth()
    rhotheta_implicit_weight_parameter = sp_init.wgt_nnew_rth()

    perturbed_theta_v_at_cells_on_half_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
    )
    nonhydro_buoy_at_cells_on_half_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, allocator=backend
    )

    cell_domain = h_grid.domain(dims.CellDim)
    start_cell_lateral_boundary_level_3 = icon_grid.start_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )

    end_cell_local = icon_grid.end_index(cell_domain(h_grid.Zone.LOCAL))

    wgtfac_c = metrics_savepoint.wgtfac_c()
    reference_theta_at_cells_on_model_levels = metrics_savepoint.theta_ref_mc()
    exner_w_explicit_weight_parameter = metrics_savepoint.vwind_expl_wgt()
    ddz_of_reference_exner_at_cells_on_half_levels = metrics_savepoint.d_exner_dz_ref_ic()
    ddqz_z_half = metrics_savepoint.ddqz_z_half()

    rho_ic_ref = sp_ref.rho_ic()
    z_theta_v_pr_ic_ref = sp_exit.z_theta_v_pr_ic()
    theta_v_ic_ref = sp_ref.theta_v_ic()
    z_th_ddz_exner_c_ref = sp_exit.z_th_ddz_exner_c()

    compute_cell_diagnostics_for_dycore.compute_interpolation_and_nonhydro_buoy.with_backend(
        backend
    )(
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
        vertical_end=icon_grid.num_levels,
        offset_provider={},
    )

    assert test_utils.dallclose(
        rho_at_cells_on_half_levels.asnumpy()[:, :], rho_ic_ref.asnumpy()[:, :]
    )

    assert test_utils.dallclose(
        theta_v_at_cells_on_half_levels.asnumpy()[:, :], theta_v_ic_ref.asnumpy()[:, :]
    )

    assert test_utils.dallclose(
        perturbed_theta_v_at_cells_on_half_levels.asnumpy()[
            start_cell_lateral_boundary_level_3:end_cell_local, 1 : icon_grid.num_levels
        ],
        z_theta_v_pr_ic_ref.asnumpy()[
            start_cell_lateral_boundary_level_3:end_cell_local, 1 : icon_grid.num_levels
        ],
        rtol=4e-9,
    )

    assert test_utils.dallclose(
        nonhydro_buoy_at_cells_on_half_levels.asnumpy()[
            start_cell_lateral_boundary_level_3:end_cell_local, 1 : icon_grid.num_levels
        ],
        z_th_ddz_exner_c_ref.asnumpy()[
            start_cell_lateral_boundary_level_3:end_cell_local, 1 : icon_grid.num_levels
        ],
        rtol=5e-10,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            definitions.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_compute_rho_theta_pgrad_and_update_vn(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment,
    step_date_init,
    step_date_exit,
    icon_grid,
    savepoint_nonhydro_init,
    is_iau_active,
    iau_wgt_dyn,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    istep_init,
    substep_init,
    substep_exit,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_exit,
    backend,
):
    sp_nh_init = savepoint_nonhydro_init
    sp_nh_exit = savepoint_nonhydro_exit
    sp_stencil_init = savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init
    sp_stencil_exit = savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_exit

    edge_domain = h_grid.domain(dims.EdgeDim)

    start_edge_lateral_boundary = icon_grid.end_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY))
    start_edge_lateral_boundary_level_7 = icon_grid.start_index(
        edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
    )
    start_edge_nudging_level_2 = icon_grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
    end_edge_nudging = icon_grid.end_index(edge_domain(h_grid.Zone.NUDGING))
    end_edge_halo = icon_grid.end_index(edge_domain(h_grid.Zone.HALO))
    end_edge_halo_level_2 = icon_grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
    end_edge_local = icon_grid.end_index(edge_domain(h_grid.Zone.LOCAL))

    vertical_config = experiment.config.vertical_grid
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)

    current_vn = sp_stencil_init.vn()
    next_vn = sp_nh_init.vn_new()
    tangential_wind = sp_stencil_init.vt()
    horizontal_pressure_gradient = sp_stencil_init.z_gradh_exner()
    perturbed_rho_at_cells_on_model_levels = sp_stencil_init.z_rth_pr(0)
    perturbed_theta_v_at_cells_on_model_levels = sp_stencil_init.z_rth_pr(1)
    hydrostatic_correction = data_alloc.zero_field(
        icon_grid, dims.EdgeDim, dims.KDim, allocator=backend
    )
    temporal_extrapolation_of_perturbed_exner = sp_stencil_init.z_exner_ex_pr()
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = (
        sp_stencil_init.z_dexner_dz_c(0)
    )
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = (
        sp_stencil_init.z_dexner_dz_c(1)
    )
    theta_v = sp_stencil_init.theta_v()
    theta_v_at_cells_on_half_levels = sp_stencil_init.theta_v_ic()
    predictor_normal_wind_advective_tendency = sp_stencil_init.ddt_vn_apc_ntl(0)
    normal_wind_tendency_due_to_slow_physics_process = sp_stencil_init.ddt_vn_phy()
    normal_wind_iau_increment = sp_stencil_init.vn_incr()
    grf_tend_vn = sp_nh_init.grf_tend_vn()
    rho_at_edges_on_model_levels = sp_stencil_init.z_rho_e()
    theta_v_at_edges_on_model_levels = sp_stencil_init.z_theta_v_e()
    config = experiment.config.nonhydrostatic
    primal_normal_cell_1 = grid_savepoint.primal_normal_cell_x()
    primal_normal_cell_2 = grid_savepoint.primal_normal_cell_y()
    dual_normal_cell_1 = grid_savepoint.dual_normal_cell_x()
    dual_normal_cell_2 = grid_savepoint.dual_normal_cell_y()

    igradp_method = config.igradp_method

    z_rho_e_ref = sp_stencil_exit.z_rho_e()
    z_theta_v_e_ref = sp_stencil_exit.z_theta_v_e()
    z_gradh_exner_ref = sp_stencil_exit.z_gradh_exner()
    vn_ref = sp_nh_exit.vn_new()

    if igradp_method == dycore_states.HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
        compute_hydrostatic_correction_term.compute_hydrostatic_correction_term.with_backend(
            backend
        )(
            theta_v=theta_v,
            ikoffset=metrics_savepoint.vertoffset_gradp(),
            zdiff_gradp=metrics_savepoint.zdiff_gradp(),
            theta_v_ic=theta_v_at_cells_on_half_levels,
            inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
            inv_dual_edge_length=grid_savepoint.inv_dual_edge_length(),
            z_hydro_corr=hydrostatic_correction,
            grav_o_cpd=constants.GRAV_O_CPD,
            horizontal_start=start_edge_nudging_level_2,
            horizontal_end=end_edge_local,
            vertical_start=icon_grid.num_levels - 1,
            vertical_end=icon_grid.num_levels,
            offset_provider={
                "E2C": icon_grid.get_connectivity("E2C"),
            },
        )
        lowest_level = icon_grid.num_levels - 1
        hydrostatic_correction_on_lowest_level = gtx.as_field(
            (dims.EdgeDim,),
            hydrostatic_correction.ndarray[:, lowest_level],
            allocator=backend,
        )
    compute_edge_diagnostics_for_dycore_and_update_vn.compute_rho_theta_pgrad_and_update_vn.with_backend(
        backend
    )(
        rho_at_edges_on_model_levels=rho_at_edges_on_model_levels,
        theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
        horizontal_pressure_gradient=horizontal_pressure_gradient,
        current_vn=current_vn,
        next_vn=next_vn,
        tangential_wind=tangential_wind,
        reference_rho_at_edges_on_model_levels=metrics_savepoint.rho_ref_me(),
        reference_theta_at_edges_on_model_levels=metrics_savepoint.theta_ref_me(),
        perturbed_rho_at_cells_on_model_levels=perturbed_rho_at_cells_on_model_levels,
        perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v_at_cells_on_model_levels,
        temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        hydrostatic_correction_on_lowest_level=hydrostatic_correction_on_lowest_level,
        predictor_normal_wind_advective_tendency=predictor_normal_wind_advective_tendency,
        normal_wind_tendency_due_to_slow_physics_process=normal_wind_tendency_due_to_slow_physics_process,
        normal_wind_iau_increment=normal_wind_iau_increment,
        grf_tend_vn=grf_tend_vn,
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        pos_on_tplane_e_x=interpolation_savepoint.pos_on_tplane_e_x(),
        pos_on_tplane_e_y=interpolation_savepoint.pos_on_tplane_e_y(),
        primal_normal_cell_x=primal_normal_cell_1,
        dual_normal_cell_x=dual_normal_cell_1,
        primal_normal_cell_y=primal_normal_cell_2,
        dual_normal_cell_y=dual_normal_cell_2,
        ddxn_z_full=metrics_savepoint.ddxn_z_full(),
        c_lin_e=interpolation_savepoint.c_lin_e(),
        ikoffset=metrics_savepoint.vertoffset_gradp(),
        zdiff_gradp=metrics_savepoint.zdiff_gradp(),
        pg_exdist=metrics_savepoint.pg_exdist_dsl(),
        inv_dual_edge_length=grid_savepoint.inv_dual_edge_length(),
        dtime=savepoint_nonhydro_init.get_metadata("dtime").get("dtime"),
        iau_wgt_dyn=iau_wgt_dyn,
        is_iau_active=is_iau_active,
        limited_area=grid_savepoint.get_metadata("limited_area").get("limited_area"),
        nflatlev=vertical_params.nflatlev,
        nflat_gradp=grid_savepoint.nflat_gradp(),
        start_edge_lateral_boundary=start_edge_lateral_boundary,
        start_edge_lateral_boundary_level_7=start_edge_lateral_boundary_level_7,
        start_edge_nudging_level_2=start_edge_nudging_level_2,
        end_edge_nudging=end_edge_nudging,
        end_edge_local=end_edge_local,
        end_edge_halo=end_edge_halo,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(end_edge_halo_level_2),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={
            "C2E2CO": icon_grid.get_connectivity("C2E2CO"),
            "E2C": icon_grid.get_connectivity("E2C"),
            "E2C2EO": icon_grid.get_connectivity("E2C2EO"),
        },
    )

    assert test_utils.dallclose(rho_at_edges_on_model_levels.asnumpy(), z_rho_e_ref.asnumpy())
    assert test_utils.dallclose(
        theta_v_at_edges_on_model_levels.asnumpy(), z_theta_v_e_ref.asnumpy()
    )

    assert test_utils.dallclose(
        horizontal_pressure_gradient.asnumpy()[start_edge_nudging_level_2:end_edge_local, :],
        z_gradh_exner_ref.asnumpy()[start_edge_nudging_level_2:end_edge_local, :],
        atol=1e-20,
    )
    assert test_utils.dallclose(
        next_vn.asnumpy()[start_edge_nudging_level_2:, :],
        vn_ref.asnumpy()[start_edge_nudging_level_2:, :],
        atol=6e-15,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep_init, substep_init, istep_exit, substep_exit",
    [(2, 1, 2, 1)],
)
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            definitions.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_apply_divergence_damping_and_update_vn(  # noqa: PLR0917 [too-many-positional-arguments]
    istep_init,
    substep_init,
    istep_exit,
    substep_exit,
    experiment,
    step_date_init,
    step_date_exit,
    icon_grid,
    savepoint_nonhydro_init,
    is_iau_active,
    iau_wgt_dyn,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_exit,
    backend,
):
    sp_nh_init = savepoint_nonhydro_init
    sp_nh_exit = savepoint_nonhydro_exit
    sp_stencil_init = savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init

    edge_domain = h_grid.domain(dims.EdgeDim)

    start_edge_nudging_level_2 = icon_grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
    end_edge_local = icon_grid.end_index(edge_domain(h_grid.Zone.LOCAL))

    dwdz_at_cells_on_model_levels = sp_stencil_init.z_dwdz_dd()
    predictor_normal_wind_advective_tendency = sp_stencil_init.ddt_vn_apc_ntl(0)
    corrector_normal_wind_advective_tendency = sp_stencil_init.ddt_vn_apc_ntl(1)
    normal_wind_tendency_due_to_slow_physics_process = sp_stencil_init.ddt_vn_phy()
    normal_wind_iau_increment = sp_stencil_init.vn_incr()
    theta_v_at_edges_on_model_levels = sp_stencil_init.z_theta_v_e()
    horizontal_pressure_gradient = sp_stencil_init.z_gradh_exner()
    current_vn = sp_stencil_init.vn()
    next_vn = savepoint_nonhydro_init.vn_new()
    horizontal_gradient_of_normal_wind_divergence = sp_nh_init.z_graddiv_vn()
    config = experiment.config.nonhydrostatic
    mean_cell_area = grid_savepoint.mean_cell_area()

    # TODO: Use serialized data ('enh_divdamp_fac' in icon) instead of computing 'interpolated_fourth_order_divdamp_factor'
    interpolated_fourth_order_divdamp_factor = data_alloc.zero_field(
        icon_grid,
        dims.KDim,
        allocator=backend,
    )

    divdamp_order = config.divdamp_order
    second_order_divdamp_scaling_coeff = sp_nh_init.divdamp_fac_o2() * mean_cell_area
    second_order_divdamp_factor = savepoint_nonhydro_init.divdamp_fac_o2()
    apply_2nd_order_divergence_damping = (
        divdamp_order == dycore_states.DivergenceDampingOrder.COMBINED
        and second_order_divdamp_scaling_coeff > 1.0e-6
    )
    apply_4th_order_divergence_damping = (
        divdamp_order == dycore_states.DivergenceDampingOrder.FOURTH_ORDER
        or (
            divdamp_order == dycore_states.DivergenceDampingOrder.COMBINED
            and second_order_divdamp_factor <= (4.0 * config.fourth_order_divdamp_factor)
        )
    )

    vn_ref = sp_nh_exit.vn_new()

    smagorinsky.en_smag_fac_for_zero_nshift.with_backend(backend)(
        grid_savepoint.vct_a(),
        config.fourth_order_divdamp_factor,
        config.fourth_order_divdamp_factor2,
        config.fourth_order_divdamp_factor3,
        config.fourth_order_divdamp_factor4,
        config.fourth_order_divdamp_z,
        config.fourth_order_divdamp_z2,
        config.fourth_order_divdamp_z3,
        config.fourth_order_divdamp_z4,
        interpolated_fourth_order_divdamp_factor,
        offset_provider={},
    )

    compute_edge_diagnostics_for_dycore_and_update_vn.apply_divergence_damping_and_update_vn.with_backend(
        backend
    )(
        horizontal_gradient_of_normal_wind_divergence=horizontal_gradient_of_normal_wind_divergence,
        next_vn=next_vn,
        current_vn=current_vn,
        dwdz_at_cells_on_model_levels=dwdz_at_cells_on_model_levels,
        predictor_normal_wind_advective_tendency=predictor_normal_wind_advective_tendency,
        corrector_normal_wind_advective_tendency=corrector_normal_wind_advective_tendency,
        normal_wind_tendency_due_to_slow_physics_process=normal_wind_tendency_due_to_slow_physics_process,
        normal_wind_iau_increment=normal_wind_iau_increment,
        theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
        horizontal_pressure_gradient=horizontal_pressure_gradient,
        second_order_divdamp_scaling_coeff=second_order_divdamp_scaling_coeff,
        horizontal_mask_for_3d_divdamp=metrics_savepoint.hmask_dd3d(),
        scaling_factor_for_3d_divdamp=metrics_savepoint.scalfac_dd3d(),
        inv_dual_edge_length=grid_savepoint.inv_dual_edge_length(),
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
        geofac_grdiv=interpolation_savepoint.geofac_grdiv(),
        advection_explicit_weight_parameter=savepoint_nonhydro_init.wgt_nnow_vel(),
        advection_implicit_weight_parameter=savepoint_nonhydro_init.wgt_nnew_vel(),
        dtime=savepoint_nonhydro_init.get_metadata("dtime").get("dtime"),
        is_iau_active=is_iau_active,
        iau_wgt_dyn=iau_wgt_dyn,
        limited_area=grid_savepoint.get_metadata("limited_area").get("limited_area"),
        apply_2nd_order_divergence_damping=apply_2nd_order_divergence_damping,
        apply_4th_order_divergence_damping=apply_4th_order_divergence_damping,
        interpolated_fourth_order_divdamp_factor=interpolated_fourth_order_divdamp_factor,
        divdamp_order=divdamp_order,
        mean_cell_area=mean_cell_area,
        second_order_divdamp_factor=second_order_divdamp_factor,
        max_nudging_coefficient=config.max_nudging_coefficient,
        dbl_eps=constants.DBL_EPS,
        horizontal_start=start_edge_nudging_level_2,
        horizontal_end=end_edge_local,
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={
            "C2E2CO": icon_grid.get_connectivity("C2E2CO"),
            "E2C": icon_grid.get_connectivity("E2C"),
            "E2C2EO": icon_grid.get_connectivity("E2C2EO"),
        },
    )

    assert test_utils.dallclose(
        next_vn.asnumpy(),
        vn_ref.asnumpy(),
        atol=4.0e-15,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            definitions.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_compute_horizontal_velocity_quantities_and_fluxes(  # noqa: PLR0917 [too-many-positional-arguments]
    istep_init,
    istep_exit,
    substep_init,
    substep_exit,
    step_date_init,
    step_date_exit,
    experiment,
    icon_grid,
    grid_savepoint,
    savepoint_dycore_30_to_38_init,
    savepoint_dycore_30_to_38_exit,
    interpolation_savepoint,
    metrics_savepoint,
    savepoint_nonhydro_init,
    savepoint_nonhydro_exit,
    backend,
):
    edge_domain = h_grid.domain(dims.EdgeDim)
    vertical_config = experiment.config.vertical_grid
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)

    ddqz_z_full_e = metrics_savepoint.ddqz_z_full_e()
    ddxn_z_full = metrics_savepoint.ddxn_z_full()
    ddxt_z_full = metrics_savepoint.ddxt_z_full()
    wgtfac_e = metrics_savepoint.wgtfac_e()
    wgtfacq_e = metrics_savepoint.wgtfacq_e()
    rbf_vec_coeff_e = interpolation_savepoint.rbf_vec_coeff_e()
    geofac_grdiv = interpolation_savepoint.geofac_grdiv()
    nflatlev = vertical_params.nflatlev

    z_vn_avg = savepoint_dycore_30_to_38_init.z_vn_avg()
    z_graddiv_vn = savepoint_dycore_30_to_38_init.z_graddiv_vn()
    vt = savepoint_dycore_30_to_38_init.vt()
    mass_fl_e = savepoint_dycore_30_to_38_init.mass_fl_e()
    z_theta_v_fl_e = savepoint_dycore_30_to_38_init.z_theta_v_fl_e()
    z_vt_ie = savepoint_dycore_30_to_38_init.z_vt_ie()
    vn_ie = savepoint_dycore_30_to_38_init.vn_ie()
    z_kin_hor_e = savepoint_dycore_30_to_38_init.z_kin_hor_e()
    z_w_concorr_me = savepoint_dycore_30_to_38_init.z_w_concorr_me()

    e_flx_avg = interpolation_savepoint.e_flx_avg()
    vn = savepoint_dycore_30_to_38_init.vn()
    z_rho_e = savepoint_dycore_30_to_38_init.z_rho_e()
    z_theta_v_e = savepoint_dycore_30_to_38_init.z_theta_v_e()

    horizontal_start = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))
    horizontal_end = icon_grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))

    z_vn_avg_ref = savepoint_dycore_30_to_38_exit.z_vn_avg()
    z_graddiv_vn_ref = savepoint_dycore_30_to_38_exit.z_graddiv_vn()
    vt_ref = savepoint_dycore_30_to_38_exit.vt()
    mass_fl_e_ref = savepoint_dycore_30_to_38_exit.mass_fl_e()
    z_theta_v_fl_e_ref = savepoint_dycore_30_to_38_exit.z_theta_v_fl_e()
    vn_ie_ref = savepoint_dycore_30_to_38_exit.vn_ie()
    z_vt_ie_ref = savepoint_dycore_30_to_38_exit.z_vt_ie()
    z_kin_hor_e_ref = savepoint_dycore_30_to_38_exit.z_kin_hor_e()
    z_w_concorr_me_ref = savepoint_dycore_30_to_38_exit.z_w_concorr_me()

    compute_horizontal_velocity_quantities.compute_horizontal_velocity_quantities_and_fluxes.with_backend(
        backend
    )(
        spatially_averaged_vn=z_vn_avg,
        horizontal_gradient_of_normal_wind_divergence=z_graddiv_vn,
        tangential_wind=vt,
        mass_flux_at_edges_on_model_levels=mass_fl_e,
        theta_v_flux_at_edges_on_model_levels=z_theta_v_fl_e,
        tangential_wind_on_half_levels=z_vt_ie,
        vn_on_half_levels=vn_ie,
        horizontal_kinetic_energy_at_edges_on_model_levels=z_kin_hor_e,
        contravariant_correction_at_edges_on_model_levels=z_w_concorr_me,
        vn=vn,
        e_flx_avg=e_flx_avg,
        geofac_grdiv=geofac_grdiv,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        rho_at_edges_on_model_levels=z_rho_e,
        theta_v_at_edges_on_model_levels=z_theta_v_e,
        ddqz_z_full_e=ddqz_z_full_e,
        ddxn_z_full=ddxn_z_full,
        ddxt_z_full=ddxt_z_full,
        wgtfac_e=wgtfac_e,
        wgtfacq_e=wgtfacq_e,
        nflatlev=nflatlev,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=0,
        vertical_end=icon_grid.num_levels + 1,
        offset_provider={
            "E2C2EO": icon_grid.get_connectivity("E2C2EO"),
            "E2C2E": icon_grid.get_connectivity("E2C2E"),
        },
    )

    assert test_utils.dallclose(
        z_vn_avg_ref.asnumpy(),
        z_vn_avg.asnumpy(),
        rtol=1.0e-6,
    )

    # same tolerances as in Liskov
    assert test_utils.dallclose(
        z_graddiv_vn_ref.asnumpy(),
        z_graddiv_vn.asnumpy(),
        rtol=1.0e-2,
        atol=1.0e-20,
    )

    assert test_utils.dallclose(
        vt_ref.asnumpy(),
        vt.asnumpy(),
        rtol=1.0e-6,
    )

    assert test_utils.dallclose(
        mass_fl_e_ref.asnumpy(),
        mass_fl_e.asnumpy(),
        rtol=1.0e-6,
    )

    assert test_utils.dallclose(
        z_theta_v_fl_e_ref.asnumpy(),
        z_theta_v_fl_e.asnumpy(),
        rtol=1.0e-6,
    )

    assert test_utils.dallclose(
        vn_ie_ref.asnumpy(),
        vn_ie.asnumpy(),
        rtol=1.0e-5,
    )

    assert test_utils.dallclose(
        z_vt_ie_ref.asnumpy(),
        z_vt_ie.asnumpy(),
        rtol=1.0e-6,
    )

    assert test_utils.dallclose(
        z_kin_hor_e_ref.asnumpy(),
        z_kin_hor_e.asnumpy(),
        rtol=1.0e-6,
    )

    assert test_utils.dallclose(
        z_w_concorr_me_ref.asnumpy(),
        z_w_concorr_me.asnumpy(),
        rtol=1.0e-7,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("at_first_substep, istep_init, istep_exit", [(True, 2, 2)])
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            definitions.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_compute_averaged_vn_and_fluxes(  # noqa: PLR0917 [too-many-positional-arguments]
    istep_init,
    istep_exit,
    substep_init,
    substep_exit,
    step_date_init,
    step_date_exit,
    experiment,
    icon_grid,
    at_first_substep,
    grid_savepoint,
    savepoint_dycore_30_to_38_init,
    savepoint_dycore_30_to_38_exit,
    interpolation_savepoint,
    metrics_savepoint,
    savepoint_nonhydro_init,
    savepoint_nonhydro_exit,
    backend,
):
    edge_domain = h_grid.domain(dims.EdgeDim)

    ddqz_z_full_e = metrics_savepoint.ddqz_z_full_e()

    z_vn_avg = savepoint_dycore_30_to_38_init.z_vn_avg()
    mass_fl_e = savepoint_dycore_30_to_38_init.mass_fl_e()
    z_theta_v_fl_e = savepoint_dycore_30_to_38_init.z_theta_v_fl_e()
    vn_traj = savepoint_nonhydro_init.vn_traj()
    mass_flx_me = savepoint_nonhydro_init.mass_flx_me()
    e_flx_avg = interpolation_savepoint.e_flx_avg()
    vn = savepoint_dycore_30_to_38_init.vn()
    z_rho_e = savepoint_dycore_30_to_38_init.z_rho_e()
    z_theta_v_e = savepoint_dycore_30_to_38_init.z_theta_v_e()
    r_nsubsteps = 1.0 / experiment.config.diffusion.ndyn_substeps

    horizontal_start = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))
    horizontal_end = icon_grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))

    z_vn_avg_ref = savepoint_dycore_30_to_38_exit.z_vn_avg()
    mass_fl_e_ref = savepoint_dycore_30_to_38_exit.mass_fl_e()
    z_theta_v_fl_e_ref = savepoint_dycore_30_to_38_exit.z_theta_v_fl_e()
    vn_traj_ref = savepoint_dycore_30_to_38_exit.vn_traj()
    mass_flx_me_ref = savepoint_dycore_30_to_38_exit.mass_flx_me()

    compute_horizontal_velocity_quantities.compute_averaged_vn_and_fluxes.with_backend(backend)(
        spatially_averaged_vn=z_vn_avg,
        mass_flux_at_edges_on_model_levels=mass_fl_e,
        theta_v_flux_at_edges_on_model_levels=z_theta_v_fl_e,
        substep_and_spatially_averaged_vn=vn_traj,
        substep_averaged_mass_flux=mass_flx_me,
        e_flx_avg=e_flx_avg,
        vn=vn,
        rho_at_edges_on_model_levels=z_rho_e,
        ddqz_z_full_e=ddqz_z_full_e,
        theta_v_at_edges_on_model_levels=z_theta_v_e,
        prepare_advection=True,
        at_first_substep=at_first_substep,
        r_nsubsteps=r_nsubsteps,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={
            "E2C2EO": icon_grid.get_connectivity("E2C2EO"),
        },
    )

    assert test_utils.dallclose(
        z_vn_avg_ref.asnumpy(),
        z_vn_avg.asnumpy(),
        rtol=1.0e-6,
    )

    assert test_utils.dallclose(
        mass_fl_e_ref.asnumpy(),
        mass_fl_e.asnumpy(),
        rtol=1.0e-6,
    )

    assert test_utils.dallclose(
        z_theta_v_fl_e_ref.asnumpy(),
        z_theta_v_fl_e.asnumpy(),
        rtol=1.0e-6,
    )

    assert test_utils.dallclose(
        vn_traj_ref.asnumpy(),
        vn_traj.asnumpy(),
        rtol=1.0e-6,
    )

    assert test_utils.dallclose(
        mass_flx_me_ref.asnumpy(),
        mass_flx_me.asnumpy(),
        rtol=1.0e-6,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("at_initial_timestep, substep_init", [(True, 1)])
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            definitions.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_vertically_implicit_solver_at_predictor_step(  # noqa: PLR0917 [too-many-positional-arguments]
    at_initial_timestep,
    substep_init,
    experiment,
    step_date_init,
    step_date_exit,
    icon_grid,
    savepoint_nonhydro_init,
    is_iau_active,
    iau_wgt_dyn,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    istep_init,
    istep_exit,
    substep_exit,
    savepoint_vertically_implicit_dycore_solver_init,
    backend,
):
    sp_nh_exit = savepoint_nonhydro_exit
    sp_stencil_init = savepoint_vertically_implicit_dycore_solver_init
    config = experiment.config.nonhydrostatic
    xp = data_alloc.import_array_ns(backend)

    vertical_config = experiment.config.vertical_grid
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)
    at_first_substep = substep_init == 1

    contravariant_correction_at_edges_on_model_levels = sp_nh_exit.z_w_concorr_me()
    mass_flux_at_edges_on_model_levels = sp_stencil_init.mass_fl_e()
    theta_v_flux_at_edges_on_model_levels = sp_stencil_init.z_theta_v_fl_e()
    predictor_vertical_wind_advective_tendency = sp_stencil_init.ddt_w_adv_pc(0)
    nonhydro_buoy_at_cells_on_half_levels = sp_stencil_init.z_th_ddz_exner_c()
    rho_at_cells_on_half_levels = sp_stencil_init.rho_ic()
    contravariant_correction_at_cells_on_half_levels = savepoint_nonhydro_init.w_concorr_c()
    current_exner = sp_stencil_init.exner_nnow()
    current_rho = sp_stencil_init.rho_nnow()
    current_theta_v = sp_stencil_init.theta_v_nnow()
    current_w = sp_stencil_init.w()
    theta_v_at_cells_on_half_levels = sp_stencil_init.theta_v_ic()
    next_w = sp_stencil_init.w()
    perturbed_exner_at_cells_on_model_levels = sp_stencil_init.exner_pr()
    exner_tendency_due_to_slow_physics = sp_stencil_init.ddt_exner_phy()
    rho_iau_increment = sp_stencil_init.rho_incr()
    exner_iau_increment = sp_stencil_init.exner_incr()
    rayleigh_damping_factor = sp_stencil_init.z_raylfac()
    next_rho = sp_stencil_init.rho()
    next_exner = sp_stencil_init.exner()
    next_theta_v = sp_stencil_init.theta_v()
    dwdz_at_cells_on_model_levels = sp_stencil_init.z_dwdz_dd()
    exner_dynamical_increment = sp_stencil_init.exner_dyn_incr()

    divdamp_type = config.divdamp_type

    w_concorr_c_ref = sp_nh_exit.w_concorr_c()
    w_ref = sp_nh_exit.w_new()
    rho_ref = sp_nh_exit.rho_new()
    exner_ref = sp_nh_exit.exner_new()
    theta_v_ref = sp_nh_exit.theta_v_new()
    z_dwdz_dd_ref = sp_nh_exit.z_dwdz_dd()
    exner_dyn_incr_ref = sp_nh_exit.exner_dyn_incr()

    geofac_div = interpolation_savepoint.geofac_div()

    cell_domain = h_grid.domain(dims.CellDim)
    start_cell_nudging = icon_grid.start_index(cell_domain(h_grid.Zone.NUDGING))
    end_cell_local = icon_grid.end_index(cell_domain(h_grid.Zone.LOCAL))
    start_cell_lateral_boundary_level_3 = icon_grid.start_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )
    end_cell_halo = icon_grid.end_index(cell_domain(h_grid.Zone.HALO))

    offset_provider = {
        "C2E": icon_grid.get_connectivity("C2E"),
    }

    vertically_implicit_dycore_solver.vertically_implicit_solver_at_predictor_step.with_backend(
        backend
    )(
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        next_w=next_w,
        next_rho=next_rho,
        next_exner=next_exner,
        next_theta_v=next_theta_v,
        dwdz_at_cells_on_model_levels=dwdz_at_cells_on_model_levels,
        exner_dynamical_increment=exner_dynamical_increment,
        geofac_div=geofac_div,
        mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
        theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
        predictor_vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
        nonhydro_buoy_at_cells_on_half_levels=nonhydro_buoy_at_cells_on_half_levels,
        rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
        contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
        exner_w_explicit_weight_parameter=metrics_savepoint.vwind_expl_wgt(),
        current_exner=current_exner,
        current_rho=current_rho,
        current_theta_v=current_theta_v,
        current_w=current_w,
        inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
        exner_w_implicit_weight_parameter=metrics_savepoint.vwind_impl_wgt(),
        theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
        perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
        exner_tendency_due_to_slow_physics=exner_tendency_due_to_slow_physics,
        rho_iau_increment=rho_iau_increment,
        exner_iau_increment=exner_iau_increment,
        ddqz_z_half=metrics_savepoint.ddqz_z_half(),
        rayleigh_damping_factor=rayleigh_damping_factor,
        reference_exner_at_cells_on_model_levels=metrics_savepoint.exner_ref_mc(),
        e_bln_c_s=interpolation_savepoint.e_bln_c_s(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        wgtfacq_c=metrics_savepoint.wgtfacq_c(),
        iau_wgt_dyn=iau_wgt_dyn,
        dtime=savepoint_nonhydro_init.get_metadata("dtime").get("dtime"),
        is_iau_active=is_iau_active,
        rayleigh_type=config.rayleigh_type,
        divdamp_type=divdamp_type,
        at_first_substep=at_first_substep,
        end_index_of_damping_layer=grid_savepoint.nrdmax(),
        kstart_moist=vertical_params.kstart_moist,
        flat_level_index_plus1=gtx.int32(vertical_params.nflatlev + 1),
        start_cell_index_nudging=start_cell_nudging,
        end_cell_index_local=end_cell_local,
        start_cell_index_lateral_lvl3=start_cell_lateral_boundary_level_3,
        end_cell_index_halo_lvl1=end_cell_halo,
        vertical_start_index_model_top=gtx.int32(0),
        vertical_end_index_model_surface=gtx.int32(icon_grid.num_levels + 1),
        offset_provider=offset_provider,
    )

    assert test_utils.dallclose(
        contravariant_correction_at_cells_on_half_levels.asnumpy(),
        w_concorr_c_ref.asnumpy(),
        atol=1e-15,
    )
    assert test_utils.dallclose(
        next_w.asnumpy()[start_cell_nudging:, :],
        w_ref.asnumpy()[start_cell_nudging:, :],
        rtol=1e-7,
        atol=1e-12,
    )
    assert test_utils.dallclose(
        next_rho.asnumpy()[start_cell_nudging:, :], rho_ref.asnumpy()[start_cell_nudging:, :]
    )
    assert test_utils.dallclose(
        next_exner.asnumpy()[start_cell_nudging:, :], exner_ref.asnumpy()[start_cell_nudging:, :]
    )
    assert test_utils.dallclose(next_theta_v.asnumpy(), theta_v_ref.asnumpy())

    # In ICON, z_dwdz_dd is computed from starting_vertical_index_for_3d_divdamp (kstart_dd3d in ICON).
    # serialized data of z_dwdz_dd can contain garbage value when k < starting_vertical_index_for_3d_divdamp.
    # Since dwdz_at_cells_on_model_levels is computed for all levels in icon4py, we have to
    # manually set the reference equal to zero when k < starting_vertical_index_for_3d_divdamp.
    starting_vertical_index_for_3d_divdamp = (
        xp.min(xp.where(metrics_savepoint.scaling_factor_for_3d_divdamp().ndarray > 0.0))[0]
        if config.divdamp_type == 32
        else 0
    )
    z_dwdz_dd_ref_with_zero_in_2d_divdamp_layers = z_dwdz_dd_ref.asnumpy()
    z_dwdz_dd_ref_with_zero_in_2d_divdamp_layers[0:starting_vertical_index_for_3d_divdamp] = 0.0
    assert test_utils.dallclose(
        dwdz_at_cells_on_model_levels.asnumpy()[start_cell_nudging:, :],
        z_dwdz_dd_ref_with_zero_in_2d_divdamp_layers[start_cell_nudging:, :],
        atol=1.0e-16,
    )

    assert test_utils.dallclose(exner_dynamical_increment.asnumpy(), exner_dyn_incr_ref.asnumpy())


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep_init, substep_init, istep_exit, substep_exit, at_initial_timestep", [(2, 1, 2, 1, True)]
)
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            definitions.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_vertically_implicit_solver_at_corrector_step(  # noqa: PLR0917 [too-many-positional-arguments]
    istep_init,
    substep_init,
    istep_exit,
    substep_exit,
    at_initial_timestep,
    experiment,
    step_date_init,
    step_date_exit,
    icon_grid,
    savepoint_nonhydro_init,
    is_iau_active,
    iau_wgt_dyn,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    savepoint_vertically_implicit_dycore_solver_init,
    backend,
):
    sp_nh_exit = savepoint_nonhydro_exit
    sp_stencil_init = savepoint_vertically_implicit_dycore_solver_init
    vertical_config = experiment.config.vertical_grid
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)

    at_first_substep = substep_init == 0
    at_last_substep = substep_exit == 0
    config = experiment.config.nonhydrostatic

    nonhydro_params = solve_nh.NonHydrostaticParams(config)

    mass_flux_at_edges_on_model_levels = sp_stencil_init.mass_fl_e()
    theta_v_flux_at_edges_on_model_levels = sp_stencil_init.z_theta_v_fl_e()
    predictor_vertical_wind_advective_tendency = sp_stencil_init.ddt_w_adv_pc(0)
    corrector_vertical_wind_advective_tendency = sp_stencil_init.ddt_w_adv_pc(1)
    nonhydro_buoy_at_cells_on_half_levels = sp_stencil_init.z_th_ddz_exner_c()
    rho_at_cells_on_half_levels = sp_stencil_init.rho_ic()
    contravariant_correction_at_cells_on_half_levels = sp_stencil_init.w_concorr_c()
    current_exner = sp_stencil_init.exner_nnow()
    current_rho = sp_stencil_init.rho_nnow()
    current_theta_v = sp_stencil_init.theta_v_nnow()
    current_w = sp_stencil_init.w()
    theta_v_at_cells_on_half_levels = sp_stencil_init.theta_v_ic()
    next_w = sp_stencil_init.w()
    perturbed_exner_at_cells_on_model_levels = sp_stencil_init.exner_pr()
    exner_tendency_due_to_slow_physics = sp_stencil_init.ddt_exner_phy()
    rho_iau_increment = sp_stencil_init.rho_incr()
    exner_iau_increment = sp_stencil_init.exner_incr()
    rayleigh_damping_factor = sp_stencil_init.z_raylfac()
    next_rho = sp_stencil_init.rho()
    next_exner = sp_stencil_init.exner()
    next_theta_v = sp_stencil_init.theta_v()
    dynamical_vertical_mass_flux_at_cells_on_half_levels = sp_stencil_init.mass_flx_ic()
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels = sp_stencil_init.vol_flx_ic()
    exner_dynamical_increment = sp_stencil_init.exner_dyn_incr()
    advection_explicit_weight_parameter = nonhydro_params.advection_explicit_weight_parameter
    advection_implicit_weight_parameter = nonhydro_params.advection_implicit_weight_parameter
    r_nsubsteps = 1.0 / experiment.config.diffusion.ndyn_substeps
    kstart_moist = vertical_params.kstart_moist

    w_ref = sp_nh_exit.w_new()
    rho_ref = sp_nh_exit.rho_new()
    exner_ref = sp_nh_exit.exner_new()
    theta_v_ref = sp_nh_exit.theta_v_new()
    exner_dyn_incr_ref = sp_nh_exit.exner_dyn_incr()
    mass_flx_ic_ref = sp_nh_exit.mass_flx_ic()
    vol_flx_ic_ref = sp_nh_exit.vol_flx_ic()

    geofac_div = interpolation_savepoint.geofac_div()

    cell_domain = h_grid.domain(dims.CellDim)
    start_cell_nudging = icon_grid.start_index(cell_domain(h_grid.Zone.NUDGING))
    end_cell_local = icon_grid.end_index(cell_domain(h_grid.Zone.LOCAL))

    offset_provider = {
        "C2E": icon_grid.get_connectivity("C2E"),
    }

    vertically_implicit_dycore_solver.vertically_implicit_solver_at_corrector_step.with_backend(
        backend
    )(
        next_w=next_w,
        next_rho=next_rho,
        next_exner=next_exner,
        next_theta_v=next_theta_v,
        dynamical_vertical_mass_flux_at_cells_on_half_levels=dynamical_vertical_mass_flux_at_cells_on_half_levels,
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
        exner_dynamical_increment=exner_dynamical_increment,
        geofac_div=geofac_div,
        mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
        theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
        predictor_vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
        corrector_vertical_wind_advective_tendency=corrector_vertical_wind_advective_tendency,
        nonhydro_buoy_at_cells_on_half_levels=nonhydro_buoy_at_cells_on_half_levels,
        rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        exner_w_explicit_weight_parameter=metrics_savepoint.vwind_expl_wgt(),
        current_exner=current_exner,
        current_rho=current_rho,
        current_theta_v=current_theta_v,
        current_w=current_w,
        inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
        exner_w_implicit_weight_parameter=metrics_savepoint.vwind_impl_wgt(),
        theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
        perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
        exner_tendency_due_to_slow_physics=exner_tendency_due_to_slow_physics,
        rho_iau_increment=rho_iau_increment,
        exner_iau_increment=exner_iau_increment,
        ddqz_z_half=metrics_savepoint.ddqz_z_half(),
        rayleigh_damping_factor=rayleigh_damping_factor,
        reference_exner_at_cells_on_model_levels=metrics_savepoint.exner_ref_mc(),
        advection_explicit_weight_parameter=advection_explicit_weight_parameter,
        advection_implicit_weight_parameter=advection_implicit_weight_parameter,
        lprep_adv=savepoint_nonhydro_init.get_metadata("prep_adv").get("prep_adv"),
        r_nsubsteps=r_nsubsteps,
        ndyn_substeps_var=float(experiment.config.diffusion.ndyn_substeps),
        iau_wgt_dyn=iau_wgt_dyn,
        dtime=savepoint_nonhydro_init.get_metadata("dtime").get("dtime"),
        is_iau_active=is_iau_active,
        rayleigh_type=config.rayleigh_type,
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
        end_index_of_damping_layer=grid_savepoint.nrdmax(),
        kstart_moist=kstart_moist,
        start_cell_index_nudging=start_cell_nudging,
        end_cell_index_local=end_cell_local,
        vertical_start_index_model_top=gtx.int32(0),
        vertical_end_index_model_surface=gtx.int32(icon_grid.num_levels + 1),
        offset_provider=offset_provider,
    )

    assert test_utils.dallclose(
        next_w.asnumpy()[start_cell_nudging:, :],
        w_ref.asnumpy()[start_cell_nudging:, :],
        rtol=1e-10,
        atol=1e-12,
    )
    assert test_utils.dallclose(
        next_rho.asnumpy()[start_cell_nudging:, :], rho_ref.asnumpy()[start_cell_nudging:, :]
    )
    assert test_utils.dallclose(
        next_exner.asnumpy()[start_cell_nudging:, :], exner_ref.asnumpy()[start_cell_nudging:, :]
    )
    assert test_utils.dallclose(next_theta_v.asnumpy(), theta_v_ref.asnumpy())
    assert test_utils.dallclose(
        dynamical_vertical_mass_flux_at_cells_on_half_levels.asnumpy()[start_cell_nudging:, :],
        mass_flx_ic_ref.asnumpy()[start_cell_nudging:, :],
        rtol=1e-10,
        atol=1e-12,
    )
    assert test_utils.dallclose(
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels.asnumpy(),
        vol_flx_ic_ref.asnumpy(),
        rtol=1e-10,
        atol=1e-12,
    )
    assert test_utils.dallclose(exner_dynamical_increment.asnumpy(), exner_dyn_incr_ref.asnumpy())

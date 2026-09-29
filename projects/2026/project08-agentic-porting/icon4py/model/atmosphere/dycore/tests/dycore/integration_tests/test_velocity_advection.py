# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next import typing as gtx_typing

from icon4py.model.atmosphere.dycore import dycore_states, velocity_advection as advection
from icon4py.model.atmosphere.dycore.stencils.compute_advection_in_horizontal_momentum_equation import (
    compute_advection_in_horizontal_momentum,
)
from icon4py.model.atmosphere.dycore.stencils.compute_advection_in_vertical_momentum_equation import (
    compute_advection_in_corrector_vertical_momentum,
    compute_advection_in_predictor_vertical_momentum,
)
from icon4py.model.atmosphere.dycore.stencils.compute_diagnostics_from_normal_wind import (
    compute_diagnostics_from_normal_wind,
)
from icon4py.model.common import dimension as dims, type_alias as ta, utils as common_utils
from icon4py.model.common.grid import (
    horizontal as h_grid,
    icon,
    states as grid_states,
    vertical as v_grid,
)
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, serialbox, test_utils

from .. import utils
from ..fixtures import *  # noqa: F403


log = logging.getLogger(__name__)


def _compare_cfl(
    *,
    vertical_cfl: np.ndarray,
    icon_result_cfl_clipping: np.ndarray,
    icon_result_max_vcfl_dyn: float,
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
) -> None:
    cfl_clipping_mask = np.where(np.abs(vertical_cfl) > 0.0, True, False)
    assert (
        cfl_clipping_mask[horizontal_start:horizontal_end, vertical_start:vertical_end]
        == icon_result_cfl_clipping[horizontal_start:horizontal_end, vertical_start:vertical_end]
    ).all()

    assert vertical_cfl[horizontal_start:horizontal_end, :].max() == icon_result_max_vcfl_dyn


def create_vertical_params(
    vertical_config: v_grid.VerticalGridConfig, grid_savepoint: serialbox.IconGridSavepoint
) -> v_grid.VerticalGrid:
    return v_grid.VerticalGrid(
        config=vertical_config, vct_a=grid_savepoint.vct_a(), vct_b=grid_savepoint.vct_b()
    )


@pytest.mark.embedded_static_args
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, step_date_init",
    [
        (definitions.Experiments.MCH_CH_R04B09, "2021-06-20T12:00:10.000"),
        (definitions.Experiments.EXCLAIM_APE, "2000-01-01T00:00:02.000"),
    ],
)
def test_verify_velocity_init_against_savepoint(  # noqa: PLR0917 [too-many-positional-arguments]
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    step_date_init: str,
    grid_savepoint: serialbox.IconGridSavepoint,
    icon_grid: icon.IconGrid,
    metrics_savepoint: serialbox.MetricSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, grid_savepoint)
    vertical_config = experiment.config.vertical_grid
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)

    velocity_advection = advection.VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=grid_savepoint.construct_edge_geometry(),
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )
    assert velocity_advection._cfl_w_limit == 0.65
    assert velocity_advection._scalfac_exdiff == 0.05
    assert test_utils.dallclose(velocity_advection._vertical_cfl.asnumpy(), 0.0)


@pytest.mark.embedded_static_args
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, step_date_init",
    [
        (definitions.Experiments.MCH_CH_R04B09, "2021-06-20T12:00:10.000"),
        (definitions.Experiments.EXCLAIM_APE, "2000-01-01T00:00:02.000"),
    ],
)
def test_scale_factors_by_dtime(  # noqa: PLR0917 [too-many-positional-arguments]
    interpolation_savepoint,
    metrics_savepoint,
    experiment,
    step_date_init,
    savepoint_velocity_init,
    icon_grid,
    grid_savepoint,
    backend,
):
    dtime = savepoint_velocity_init.get_metadata("dtime").get("dtime")
    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, grid_savepoint)
    vertical_config = experiment.config.vertical_grid
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)

    velocity_advection = advection.VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=grid_savepoint.construct_edge_geometry(),
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )
    (cfl_w_limit, scalfac_exdiff) = velocity_advection._scale_factors_by_dtime(dtime)
    assert cfl_w_limit == savepoint_velocity_init.cfl_w_limit()
    assert scalfac_exdiff == savepoint_velocity_init.scalfac_exdiff()


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
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:20.000",
            "2021-06-20T12:00:20.000",
        ),
        (definitions.Experiments.EXCLAIM_APE, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
def test_velocity_predictor_step(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment,
    step_date_init,
    step_date_exit,
    icon_grid,
    grid_savepoint,
    savepoint_velocity_init,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_velocity_exit,
    backend,
    caplog,
):
    caplog.set_level(logging.WARN)
    init_savepoint = savepoint_velocity_init
    vn_only = init_savepoint.vn_only()
    dtime = init_savepoint.get_metadata("dtime").get("dtime")

    diagnostic_state = dycore_states.DiagnosticStateNonHydro(
        max_vertical_cfl=data_alloc.scalar_like_array(0.0, backend),
        tangential_wind=init_savepoint.vt(),
        vn_on_half_levels=init_savepoint.vn_ie(),
        contravariant_correction_at_cells_on_half_levels=init_savepoint.w_concorr_c(),
        theta_v_at_cells_on_half_levels=None,
        perturbed_exner_at_cells_on_model_levels=None,
        rho_at_cells_on_half_levels=None,
        exner_tendency_due_to_slow_physics=None,
        grf_tend_rho=None,
        grf_tend_thv=None,
        grf_tend_w=None,
        mass_flux_at_edges_on_model_levels=None,
        normal_wind_tendency_due_to_slow_physics_process=None,
        grf_tend_vn=None,
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_vn_apc_pc(0), init_savepoint.ddt_vn_apc_pc(1)
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_w_adv_pc(0), init_savepoint.ddt_w_adv_pc(1)
        ),
        rho_iau_increment=None,
        normal_wind_iau_increment=None,
        exner_iau_increment=None,
        exner_dynamical_increment=None,
    )
    prognostic_state = prognostics.PrognosticState(
        w=init_savepoint.w(),
        vn=init_savepoint.vn(),
        theta_v=None,
        rho=None,
        exner=None,
    )
    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, grid_savepoint)

    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()

    vertical_config = experiment.config.vertical_grid
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)

    velocity_advection = advection.VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=edge_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )

    velocity_advection.run_predictor_step(
        skip_compute_predictor_vertical_advection=vn_only,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        contravariant_correction_at_edges_on_model_levels=init_savepoint.z_w_concorr_me(),
        horizontal_kinetic_energy_at_edges_on_model_levels=init_savepoint.z_kin_hor_e(),
        tangential_wind_on_half_levels=init_savepoint.z_vt_ie(),
        dtime=dtime,
        cell_areas=cell_geometry.area,
    )

    icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc(0).asnumpy()
    icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc(0).asnumpy()
    icon_result_vn_ie = savepoint_velocity_exit.vn_ie().asnumpy()
    icon_result_vt = savepoint_velocity_exit.vt().asnumpy()
    icon_result_w_concorr_c = savepoint_velocity_exit.w_concorr_c().asnumpy()
    icon_result_max_vcfl_dyn = savepoint_velocity_exit.max_vcfl_dyn()

    assert test_utils.dallclose(
        diagnostic_state.tangential_wind.asnumpy(), icon_result_vt, atol=1.0e-14
    )

    assert test_utils.dallclose(
        diagnostic_state.vn_on_half_levels.asnumpy(), icon_result_vn_ie, atol=1.0e-14
    )

    start_cell_nudging = icon_grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING))
    assert test_utils.dallclose(
        diagnostic_state.contravariant_correction_at_cells_on_half_levels.asnumpy()[
            start_cell_nudging:, vertical_params.nflatlev + 1 : icon_grid.num_levels
        ],
        icon_result_w_concorr_c[
            start_cell_nudging:, vertical_params.nflatlev + 1 : icon_grid.num_levels
        ],
        atol=1.0e-15,
    )

    assert test_utils.dallclose(
        diagnostic_state.vertical_wind_advective_tendency.predictor.asnumpy()[
            start_cell_nudging:, :
        ],
        icon_result_ddt_w_adv_pc[start_cell_nudging:, :],
        atol=5.0e-16,
        rtol=1.0e-10,
    )

    assert test_utils.dallclose(
        diagnostic_state.normal_wind_advective_tendency.predictor.asnumpy(),
        icon_result_ddt_vn_apc_pc,
        atol=1.0e-15,
    )

    assert diagnostic_state.max_vertical_cfl == icon_result_max_vcfl_dyn


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("istep_init, istep_exit", [(2, 2)])
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:20.000",
            "2021-06-20T12:00:20.000",
        ),
        (definitions.Experiments.EXCLAIM_APE, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
def test_velocity_corrector_step(  # noqa: PLR0917 [too-many-positional-arguments]
    istep_init,
    istep_exit,
    experiment,
    step_date_init,
    step_date_exit,
    icon_grid,
    grid_savepoint,
    savepoint_velocity_init,
    savepoint_velocity_exit,
    interpolation_savepoint,
    metrics_savepoint,
    backend,
):
    init_savepoint = savepoint_velocity_init
    vn_only = init_savepoint.vn_only()
    dtime = init_savepoint.get_metadata("dtime").get("dtime")

    assert not vn_only

    diagnostic_state = dycore_states.DiagnosticStateNonHydro(
        max_vertical_cfl=data_alloc.scalar_like_array(0.0, backend),
        tangential_wind=init_savepoint.vt(),
        vn_on_half_levels=init_savepoint.vn_ie(),
        contravariant_correction_at_cells_on_half_levels=init_savepoint.w_concorr_c(),
        theta_v_at_cells_on_half_levels=None,
        perturbed_exner_at_cells_on_model_levels=None,
        rho_at_cells_on_half_levels=None,
        exner_tendency_due_to_slow_physics=None,
        grf_tend_rho=None,
        grf_tend_thv=None,
        grf_tend_w=None,
        mass_flux_at_edges_on_model_levels=None,
        normal_wind_tendency_due_to_slow_physics_process=None,
        grf_tend_vn=None,
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_vn_apc_pc(0), init_savepoint.ddt_vn_apc_pc(1)
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            init_savepoint.ddt_w_adv_pc(0), init_savepoint.ddt_w_adv_pc(1)
        ),
        rho_iau_increment=None,
        normal_wind_iau_increment=None,
        exner_iau_increment=None,  # sp.exner_incr(),
        exner_dynamical_increment=None,
    )
    prognostic_state = prognostics.PrognosticState(
        w=init_savepoint.w(),
        vn=init_savepoint.vn(),
        theta_v=None,
        rho=None,
        exner=None,
    )

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)

    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, grid_savepoint)

    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()

    vertical_config = experiment.config.vertical_grid
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)

    velocity_advection = advection.VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_params=edge_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )

    velocity_advection.run_corrector_step(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        horizontal_kinetic_energy_at_edges_on_model_levels=init_savepoint.z_kin_hor_e(),
        tangential_wind_on_half_levels=init_savepoint.z_vt_ie(),
        dtime=dtime,
        cell_areas=cell_geometry.area,
    )

    icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc(1).asnumpy()
    icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc(1).asnumpy()
    icon_result_max_vcfl_dyn = savepoint_velocity_exit.max_vcfl_dyn()

    start_cell_nudging = icon_grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING))
    assert test_utils.dallclose(
        diagnostic_state.vertical_wind_advective_tendency.corrector.asnumpy()[
            start_cell_nudging:, :
        ],
        icon_result_ddt_w_adv_pc[start_cell_nudging:, :],
        atol=5.0e-16,
    )
    assert test_utils.dallclose(
        diagnostic_state.normal_wind_advective_tendency.corrector.asnumpy(),
        icon_result_ddt_vn_apc_pc,
        atol=5.0e-16,
    )

    assert diagnostic_state.max_vertical_cfl == icon_result_max_vcfl_dyn


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (definitions.Experiments.EXCLAIM_APE, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
def test_compute_diagnostics_from_normal_wind(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment,
    step_date_init,
    step_date_exit,
    icon_grid,
    grid_savepoint,
    interpolation_savepoint,
    metrics_savepoint,
    savepoint_velocity_init,
    savepoint_velocity_exit,
    backend,
):
    edge_domain = h_grid.domain(dims.EdgeDim)

    tangential_wind_on_half_levels = savepoint_velocity_init.z_vt_ie()
    tangential_wind = savepoint_velocity_init.vt()
    vn_on_half_levels = savepoint_velocity_init.vn_ie()
    horizontal_kinetic_energy_at_edges_on_model_levels = savepoint_velocity_init.z_kin_hor_e()
    horizontal_advection_of_w_at_edges_on_half_levels = data_alloc.zero_field(
        icon_grid, dims.EdgeDim, dims.KDim, allocator=backend
    )
    vn = savepoint_velocity_init.vn()
    w = savepoint_velocity_init.w()

    rbf_vec_coeff_e = interpolation_savepoint.rbf_vec_coeff_e()
    wgtfac_e = metrics_savepoint.wgtfac_e()
    ddxn_z_full = metrics_savepoint.ddxn_z_full()
    ddxt_z_full = metrics_savepoint.ddxt_z_full()
    contravariant_correction_at_edges_on_model_levels = savepoint_velocity_init.z_w_concorr_me()
    wgtfacq_e = metrics_savepoint.wgtfacq_e()
    nflatlev = grid_savepoint.nflatlev()
    c_intp = interpolation_savepoint.c_intp()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
    tangent_orientation = grid_savepoint.tangent_orientation()

    skip_compute_predictor_vertical_advection = savepoint_velocity_init.vn_only()
    # TODO(havogt): we need a test where skip_compute_predictor_vertical_advection is True!

    horizontal_start = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))
    horizontal_end = icon_grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))

    icon_result_vt = savepoint_velocity_exit.vt()
    icon_result_z_vt_ie = savepoint_velocity_exit.z_vt_ie()
    icon_result_vn_ie = savepoint_velocity_exit.vn_ie()
    icon_result_z_kin_hor_e = savepoint_velocity_exit.z_kin_hor_e()
    icon_result_z_w_concorr_me = savepoint_velocity_exit.z_w_concorr_me()
    icon_result_z_v_grad_w = savepoint_velocity_exit.z_v_grad_w()

    compute_diagnostics_from_normal_wind.with_backend(backend)(
        tangential_wind=tangential_wind,
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn_on_half_levels=vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
        horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
        vn=vn,
        w=w,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        wgtfac_e=wgtfac_e,
        ddxn_z_full=ddxn_z_full,
        ddxt_z_full=ddxt_z_full,
        wgtfacq_e=wgtfacq_e,
        c_intp=c_intp,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
        nflatlev=gtx.int32(nflatlev),
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels + 1),
        offset_provider={
            "E2C": icon_grid.get_connectivity("E2C"),
            "E2V": icon_grid.get_connectivity("E2V"),
            "V2C": icon_grid.get_connectivity("V2C"),
            "E2C2E": icon_grid.get_connectivity("E2C2E"),
        },
    )

    assert test_utils.dallclose(
        icon_result_vt.asnumpy(), tangential_wind.asnumpy(), rtol=1.0e-14, atol=1.0e-14
    )
    assert test_utils.dallclose(
        icon_result_z_vt_ie.asnumpy(),
        tangential_wind_on_half_levels.asnumpy(),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert test_utils.dallclose(
        icon_result_vn_ie.asnumpy(), vn_on_half_levels.asnumpy(), rtol=1.0e-15, atol=1.0e-15
    )
    assert test_utils.dallclose(
        icon_result_z_kin_hor_e.asnumpy(),
        horizontal_kinetic_energy_at_edges_on_model_levels.asnumpy(),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert test_utils.dallclose(
        icon_result_z_w_concorr_me.asnumpy(),
        contravariant_correction_at_edges_on_model_levels.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    # the restriction is ok, as this is a velocity advection temporary
    lateral_boundary_7 = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7))
    halo_1 = icon_grid.end_index(edge_domain(h_grid.Zone.HALO))
    assert test_utils.dallclose(
        icon_result_z_v_grad_w.asnumpy()[lateral_boundary_7:halo_1, :],
        horizontal_advection_of_w_at_edges_on_half_levels.asnumpy()[lateral_boundary_7:halo_1, :],
        rtol=1.0e-15,
        atol=1.0e-15,
    )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:20.000",
            "2021-06-20T12:00:20.000",
        ),
        (definitions.Experiments.EXCLAIM_APE, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
@pytest.mark.parametrize("istep_init, istep_exit", [(1, 1)])
def test_compute_advection_in_predictor_vertical_momentum(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment,
    step_date_init,
    step_date_exit,
    istep_init,
    istep_exit,
    icon_grid,
    grid_savepoint,
    interpolation_savepoint,
    metrics_savepoint,
    savepoint_velocity_exit,
    backend,
    savepoint_velocity_init,
):
    scalfac_exdiff = savepoint_velocity_init.scalfac_exdiff()
    cfl_w_limit = savepoint_velocity_init.cfl_w_limit()
    ddqz_z_half = metrics_savepoint.ddqz_z_half()
    contravariant_correction_at_edges_on_model_levels = savepoint_velocity_exit.z_w_concorr_me()
    contravariant_correction_at_cells_on_half_levels = savepoint_velocity_init.w_concorr_c()
    w = savepoint_velocity_init.w()
    horizontal_advection_of_w_at_edges_on_half_levels = savepoint_velocity_exit.z_v_grad_w()
    vertical_wind_advective_tendency = savepoint_velocity_init.ddt_w_adv_pc(istep_init - 1)
    contravariant_corrected_w_at_cells_on_model_levels = savepoint_velocity_init.z_w_con_c_full()
    vertical_cfl = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat, allocator=backend
    )
    skip_compute_predictor_vertical_advection = savepoint_velocity_init.lvn_only()

    coeff1_dwdz = metrics_savepoint.coeff1_dwdz()
    coeff2_dwdz = metrics_savepoint.coeff2_dwdz()
    e_bln_c_s = interpolation_savepoint.e_bln_c_s()
    wgtfac_c = metrics_savepoint.wgtfac_c()
    owner_mask = grid_savepoint.c_owner_mask()
    area = grid_savepoint.cell_areas()
    geofac_n2s = interpolation_savepoint.geofac_n2s()

    icon_result_z_w_con_c_full = savepoint_velocity_exit.z_w_con_c_full()
    icon_result_ddt_w_adv = savepoint_velocity_exit.ddt_w_adv_pc(istep_exit - 1)
    icon_result_w_concorr_c = savepoint_velocity_exit.w_concorr_c()
    icon_result_cfl_clipping = savepoint_velocity_exit.cfl_clipping()
    icon_result_max_vcfl_dyn = savepoint_velocity_exit.max_vcfl_dyn()

    end_index_of_damping_layer = grid_savepoint.nrdmax()

    dtime = savepoint_velocity_init.get_metadata("dtime").get("dtime")
    cell_domain = h_grid.domain(dims.CellDim)
    start_cell_nudging_for_vertical_wind_advective_tendency = icon_grid.start_index(
        cell_domain(h_grid.Zone.NUDGING)
    )
    end_cell_local_for_vertical_wind_advective_tendency = icon_grid.end_index(
        cell_domain(h_grid.Zone.LOCAL)
    )
    horizontal_start = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
    horizontal_end = icon_grid.end_index(cell_domain(h_grid.Zone.HALO))
    vertical_start = 0
    vertical_end = icon_grid.num_levels
    compute_advection_in_predictor_vertical_momentum.with_backend(backend)(
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
        nflatlev=grid_savepoint.nflatlev(),
        end_index_of_damping_layer=end_index_of_damping_layer,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={
            "C2E": icon_grid.get_connectivity("C2E"),
            "C2E2CO": icon_grid.get_connectivity("C2E2CO"),
            "V2C": icon_grid.get_connectivity("V2C"),
            "E2C": icon_grid.get_connectivity("E2C"),
            "E2V": icon_grid.get_connectivity("E2V"),
        },
    )

    assert test_utils.dallclose(
        icon_result_w_concorr_c.asnumpy(),
        contravariant_correction_at_cells_on_half_levels.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )

    assert test_utils.dallclose(
        icon_result_z_w_con_c_full.asnumpy(),
        contravariant_corrected_w_at_cells_on_model_levels.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    assert test_utils.dallclose(
        icon_result_ddt_w_adv.asnumpy()[
            start_cell_nudging_for_vertical_wind_advective_tendency:end_cell_local_for_vertical_wind_advective_tendency,
            :,
        ],
        vertical_wind_advective_tendency.asnumpy()[
            start_cell_nudging_for_vertical_wind_advective_tendency:end_cell_local_for_vertical_wind_advective_tendency,
            :,
        ],
        rtol=1.0e-15,
        atol=1.0e-15,
    )

    # TODO(OngChia): currently direct comparison of vcfl_dsl is not possible because it is not properly updated in icon run
    _compare_cfl(
        vertical_cfl=vertical_cfl.asnumpy(),
        icon_result_cfl_clipping=icon_result_cfl_clipping.asnumpy(),
        icon_result_max_vcfl_dyn=icon_result_max_vcfl_dyn,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=max(2, end_index_of_damping_layer - 2),
        vertical_end=icon_grid.num_levels - 3,
    )


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:20.000",
            "2021-06-20T12:00:20.000",
        ),
        (definitions.Experiments.EXCLAIM_APE, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
@pytest.mark.parametrize("istep_init, istep_exit", [(2, 2)])
def test_compute_advection_in_corrector_vertical_momentum(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment,
    step_date_init,
    step_date_exit,
    istep_init,
    istep_exit,
    icon_grid,
    grid_savepoint,
    interpolation_savepoint,
    metrics_savepoint,
    savepoint_velocity_exit,
    savepoint_velocity_init,
    backend,
):
    scalfac_exdiff = savepoint_velocity_init.scalfac_exdiff()
    cfl_w_limit = savepoint_velocity_init.cfl_w_limit()
    ddqz_z_half = metrics_savepoint.ddqz_z_half()
    contravariant_correction_at_cells_on_half_levels = savepoint_velocity_exit.w_concorr_c()
    w = savepoint_velocity_init.w()
    tangential_wind_on_half_levels = savepoint_velocity_exit.z_vt_ie()
    vn_on_half_levels = savepoint_velocity_exit.vn_ie()
    vertical_wind_advective_tendency = savepoint_velocity_init.ddt_w_adv_pc(istep_init - 1)
    contravariant_corrected_w_at_cells_on_model_levels = savepoint_velocity_init.z_w_con_c_full()
    vertical_cfl = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat, allocator=backend
    )

    coeff1_dwdz = metrics_savepoint.coeff1_dwdz()
    coeff2_dwdz = metrics_savepoint.coeff2_dwdz()
    c_intp = interpolation_savepoint.c_intp()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
    tangent_orientation = grid_savepoint.tangent_orientation()
    e_bln_c_s = interpolation_savepoint.e_bln_c_s()
    owner_mask = grid_savepoint.c_owner_mask()
    area = grid_savepoint.cell_areas()
    geofac_n2s = interpolation_savepoint.geofac_n2s()

    icon_result_z_w_con_c_full = savepoint_velocity_exit.z_w_con_c_full()
    icon_result_ddt_w_adv = savepoint_velocity_exit.ddt_w_adv_pc(istep_exit - 1)
    icon_result_cfl_clipping = savepoint_velocity_exit.cfl_clipping()
    icon_result_max_vcfl_dyn = savepoint_velocity_exit.max_vcfl_dyn()

    end_index_of_damping_layer = grid_savepoint.nrdmax()

    dtime = savepoint_velocity_init.get_metadata("dtime").get("dtime")
    cell_domain = h_grid.domain(dims.CellDim)
    start_cell_nudging_for_vertical_wind_advective_tendency = icon_grid.start_index(
        cell_domain(h_grid.Zone.NUDGING)
    )
    end_cell_local_for_vertical_wind_advective_tendency = icon_grid.end_index(
        cell_domain(h_grid.Zone.LOCAL)
    )
    horizontal_start = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
    horizontal_end = icon_grid.end_index(cell_domain(h_grid.Zone.HALO))
    vertical_start = 0
    vertical_end = icon_grid.num_levels
    compute_advection_in_corrector_vertical_momentum.with_backend(backend)(
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
        offset_provider={
            "C2E": icon_grid.get_connectivity("C2E"),
            "C2E2CO": icon_grid.get_connectivity("C2E2CO"),
            "V2C": icon_grid.get_connectivity("V2C"),
            "E2C": icon_grid.get_connectivity("E2C"),
            "E2V": icon_grid.get_connectivity("E2V"),
        },
    )

    assert test_utils.dallclose(
        icon_result_z_w_con_c_full.asnumpy(),
        contravariant_corrected_w_at_cells_on_model_levels.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    assert test_utils.dallclose(
        icon_result_ddt_w_adv.asnumpy()[
            start_cell_nudging_for_vertical_wind_advective_tendency:end_cell_local_for_vertical_wind_advective_tendency,
            :,
        ],
        vertical_wind_advective_tendency.asnumpy()[
            start_cell_nudging_for_vertical_wind_advective_tendency:end_cell_local_for_vertical_wind_advective_tendency,
            :,
        ],
        rtol=1.0e-15,
        atol=1.0e-15,
    )

    # TODO(OngChia): currently direct comparison of vcfl_dsl is not possible because it is not properly updated in icon run
    _compare_cfl(
        vertical_cfl=vertical_cfl.asnumpy(),
        icon_result_cfl_clipping=icon_result_cfl_clipping.asnumpy(),
        icon_result_max_vcfl_dyn=icon_result_max_vcfl_dyn,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=max(2, end_index_of_damping_layer - 2),
        vertical_end=icon_grid.num_levels - 3,
    )


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (definitions.Experiments.EXCLAIM_APE, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
@pytest.mark.parametrize("istep_init, istep_exit", [(1, 1), (2, 2)])
def test_compute_advection_in_horizontal_momentum(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment,
    step_date_init,
    step_date_exit,
    istep_init,
    istep_exit,
    icon_grid,
    grid_savepoint,
    interpolation_savepoint,
    metrics_savepoint,
    backend,
    savepoint_velocity_init,
    savepoint_velocity_exit,
):
    vn = savepoint_velocity_init.vn()
    horizontal_kinetic_energy_at_edges_on_model_levels = savepoint_velocity_exit.z_kin_hor_e()
    tangential_wind = savepoint_velocity_exit.vt()
    contravariant_corrected_w_at_cells_on_model_levels = savepoint_velocity_exit.z_w_con_c_full()
    vn_on_half_levels = savepoint_velocity_exit.vn_ie()
    normal_wind_advective_tendency = savepoint_velocity_init.ddt_vn_apc_pc(istep_init - 1)

    e_bln_c_s = interpolation_savepoint.e_bln_c_s()
    geofac_rot = interpolation_savepoint.geofac_rot()
    coeff_gradekin = metrics_savepoint.coeff_gradekin()
    coriolis_frequency = grid_savepoint.f_e()
    c_lin_e = interpolation_savepoint.c_lin_e()
    ddqz_z_full_e = metrics_savepoint.ddqz_z_full_e()
    area_edge = grid_savepoint.edge_areas()
    tangent_orientation = grid_savepoint.tangent_orientation()
    inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
    geofac_grdiv = interpolation_savepoint.geofac_grdiv()

    edge_domain = h_grid.domain(dims.EdgeDim)

    start_edge_nudging_level_2 = icon_grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
    end_edge_local = icon_grid.end_index(edge_domain(h_grid.Zone.LOCAL))

    dtime = savepoint_velocity_init.get_metadata("dtime").get("dtime")
    end_index_of_damping_layer = grid_savepoint.nrdmax()

    icon_result_ddt_vn_apc = savepoint_velocity_exit.ddt_vn_apc_pc(istep_exit - 1)

    scalfac_exdiff = savepoint_velocity_init.scalfac_exdiff()
    cfl_w_limit = savepoint_velocity_init.cfl_w_limit()
    max_vertical_cfl = savepoint_velocity_exit.max_vcfl_dyn()
    apply_extra_diffusion_on_vn = max_vertical_cfl > cfl_w_limit * dtime

    compute_advection_in_horizontal_momentum.with_backend(backend)(
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
        horizontal_start=start_edge_nudging_level_2,
        horizontal_end=end_edge_local,
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={
            "V2E": icon_grid.get_connectivity("V2E"),
            "E2V": icon_grid.get_connectivity("E2V"),
            "E2C": icon_grid.get_connectivity("E2C"),
            "E2C2EO": icon_grid.get_connectivity("E2C2EO"),
            "C2E": icon_grid.get_connectivity("C2E"),
        },
    )

    assert test_utils.dallclose(
        icon_result_ddt_vn_apc.asnumpy(),
        normal_wind_advective_tendency.asnumpy(),
        rtol=1.0e-15,
        atol=1.0e-15,
    )

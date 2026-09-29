# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import collections

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states, diffusion_utils
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import geometry_attributes as geometry_meta, vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    definitions,
    grid_utils,
    reference_funcs as ref_funcs,
    serialbox as sb,
    test_utils,
)

from ..fixtures import *  # noqa: F403
from ..utils import diff_multfac_vn_numpy, smag_limit_numpy, verify_diffusion_fields


grid_functionality = collections.defaultdict(dict)


def get_grid_for_experiment(experiment: definitions.Experiment, backend: gtx_typing.Backend):
    return _get_or_initialize(experiment, backend, "grid")


def get_edge_geometry_for_experiment(
    experiment: definitions.Experiment, backend: gtx_typing.Backend
):
    return _get_or_initialize(experiment, backend, "edge_geometry")


def get_cell_geometry_for_experiment(
    experiment: definitions.Experiment, backend: gtx_typing.Backend
):
    return _get_or_initialize(experiment, backend, "cell_geometry")


def _get_or_initialize(experiment: definitions.Experiment, backend: gtx_typing.Backend, name: str):
    if not grid_functionality[experiment.name].get(name):
        geometry_ = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
        grid = geometry_.grid

        cell_params = grid_states.CellParams(
            cell_center_lat=geometry_.get(geometry_meta.CELL_LAT),
            cell_center_lon=geometry_.get(geometry_meta.CELL_LON),
            area=geometry_.get(geometry_meta.CELL_AREA),
        )
        edge_params = grid_states.EdgeParams(
            edge_center_lat=geometry_.get(geometry_meta.EDGE_LAT),
            edge_center_lon=geometry_.get(geometry_meta.EDGE_LON),
            tangent_orientation=geometry_.get(geometry_meta.TANGENT_ORIENTATION),
            coriolis_frequency=geometry_.get(geometry_meta.CORIOLIS_PARAMETER),
            edge_areas=geometry_.get(geometry_meta.EDGE_AREA),
            primal_edge_lengths=geometry_.get(geometry_meta.EDGE_LENGTH),
            inverse_primal_edge_lengths=geometry_.get(f"inverse_of_{geometry_meta.EDGE_LENGTH}"),
            dual_edge_lengths=geometry_.get(geometry_meta.DUAL_EDGE_LENGTH),
            inverse_dual_edge_lengths=geometry_.get(f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}"),
            inverse_vertex_vertex_lengths=geometry_.get(
                f"inverse_of_{geometry_meta.VERTEX_VERTEX_LENGTH}"
            ),
            primal_normal_x=geometry_.get(geometry_meta.EDGE_NORMAL_U),
            primal_normal_y=geometry_.get(geometry_meta.EDGE_NORMAL_V),
            primal_normal_cell_x=geometry_.get(geometry_meta.EDGE_NORMAL_CELL_U),
            primal_normal_cell_y=geometry_.get(geometry_meta.EDGE_NORMAL_CELL_V),
            primal_normal_vert_x=geometry_.get(geometry_meta.EDGE_NORMAL_VERTEX_U),
            primal_normal_vert_y=geometry_.get(geometry_meta.EDGE_NORMAL_VERTEX_V),
            dual_normal_cell_x=geometry_.get(geometry_meta.EDGE_TANGENT_CELL_U),
            dual_normal_cell_y=geometry_.get(geometry_meta.EDGE_TANGENT_CELL_V),
            dual_normal_vert_x=geometry_.get(geometry_meta.EDGE_TANGENT_VERTEX_U),
            dual_normal_vert_y=geometry_.get(geometry_meta.EDGE_TANGENT_VERTEX_V),
        )
        grid_functionality[experiment.name]["grid"] = grid
        grid_functionality[experiment.name]["edge_geometry"] = edge_params
        grid_functionality[experiment.name]["cell_geometry"] = cell_params
    return grid_functionality[experiment.name].get(name)


def test_diffusion_coefficients_with_hdiff_efdt_ratio():
    config = diffusion.DiffusionConfig()
    config.hdiff_efdt_ratio = 1.0
    config.hdiff_w_efdt_ratio = 2.0

    params = diffusion.DiffusionParams(config)

    assert pytest.approx(0.125, abs=1e-12) == params.K2
    assert pytest.approx(0.125 / 8.0, abs=1e-12) == params.K4
    assert pytest.approx(0.125 / 64.0, abs=1e-12) == params.K6
    assert pytest.approx(1.0 / 72.0, abs=1e-12) == params.K4W


def test_diffusion_coefficients_without_hdiff_efdt_ratio():
    config = diffusion.DiffusionConfig()
    config.hdiff_efdt_ratio = 0.0
    config.hdiff_w_efdt_ratio = 0.0

    params = diffusion.DiffusionParams(config)

    assert params.K2 == 0.0
    assert params.K4 == 0.0
    assert params.K6 == 0.0
    assert params.K4W == 0.0


def test_smagorinski_heights_diffusion_type_5_are_consistent():
    config = diffusion.DiffusionConfig()
    config.smagorinski_scaling_factor = 0.15
    config.diffusion_type = 5

    params = diffusion.DiffusionParams(config)
    assert len(params.smagorinski_height) == 4
    assert min(params.smagorinski_height) == params.smagorinski_height[0]
    assert max(params.smagorinski_height) == params.smagorinski_height[-1]
    assert params.smagorinski_height[0] < params.smagorinski_height[1]
    assert params.smagorinski_height[1] < params.smagorinski_height[3]
    assert params.smagorinski_height[2] != params.smagorinski_height[1]
    assert params.smagorinski_height[2] != params.smagorinski_height[3]


def test_smagorinski_factor_diffusion_type_5():
    params = diffusion.DiffusionParams(diffusion.DiffusionConfig())
    assert len(params.smagorinski_factor) == len(params.smagorinski_height)
    assert len(params.smagorinski_factor) == 4
    assert all(p >= 0 for p in params.smagorinski_factor)


@pytest.mark.uses_concat_where
@pytest.mark.datatest
# TODO(havogt): Remove custom `experiment` parametrization
@pytest.mark.parametrize(
    "experiment_description,step_date_init",
    [
        (definitions.Experiments.MCH_CH_R04B09, "2021-06-20T12:00:10.000"),
        (definitions.Experiments.MCH_CH_R04B09, "2021-06-20T12:00:20.000"),
    ],
)
def test_diffusion_init(  # noqa: PLR0917 [too-many-positional-arguments]
    savepoint_diffusion_init,
    interpolation_state: diffusion_states.DiffusionInterpolationState,
    metric_state: diffusion_states.DiffusionMetricState,
    experiment,
    step_date_init,
    backend,
):
    config = experiment.config.diffusion
    additional_parameters = diffusion.DiffusionParams(config)

    grid = get_grid_for_experiment(experiment, backend)
    cell_params = get_cell_geometry_for_experiment(experiment, backend)
    edge_params = get_edge_geometry_for_experiment(experiment, backend)

    vertical_config = experiment.config.vertical_grid
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )

    meta = savepoint_diffusion_init.get_metadata("linit", "date")

    assert meta["linit"] is False
    assert meta["date"] == step_date_init

    diffusion_granule = diffusion.Diffusion(
        grid=grid,
        config=config,
        params=additional_parameters,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
        backend=backend,
        exchange=decomp_defs.single_node_exchange,
    )

    assert diffusion_granule.diff_multfac_w == min(
        1.0 / 48.0, additional_parameters.K4W * config.substep_as_float
    )

    assert test_utils.dallclose(diffusion_granule.v_vert.asnumpy(), 0.0)
    assert test_utils.dallclose(diffusion_granule.u_vert.asnumpy(), 0.0)
    assert test_utils.dallclose(diffusion_granule.kh_smag_ec.asnumpy(), 0.0)
    assert test_utils.dallclose(diffusion_granule.kh_smag_e.asnumpy(), 0.0)

    shape_k = (grid.num_levels,)
    expected_smag_limit = smag_limit_numpy(
        diff_multfac_vn_numpy,
        shape_k,
        additional_parameters.K4,
        config.substep_as_float,
    )

    assert (
        diffusion_granule.smag_offset == 0.25 * additional_parameters.K4 * config.substep_as_float
    )
    assert test_utils.dallclose(diffusion_granule.smag_limit.asnumpy(), expected_smag_limit)

    expected_diff_multfac_vn = diff_multfac_vn_numpy(
        shape_k, additional_parameters.K4, config.substep_as_float
    )

    assert test_utils.dallclose(
        diffusion_granule.diff_multfac_vn.asnumpy(), expected_diff_multfac_vn
    )
    expected_enh_smag_fac = ref_funcs.enhanced_smagorinski_factor_numpy(
        additional_parameters.smagorinski_factor,
        additional_parameters.smagorinski_height,
        vertical_params.vct_a.ndarray,
    )
    assert test_utils.dallclose(diffusion_granule.enh_smag_fac.asnumpy(), expected_enh_smag_fac)


def _verify_init_values_against_savepoint(
    savepoint: sb.IconDiffusionInitSavepoint, diffusion_granule: diffusion.Diffusion, backend
):
    dtime = savepoint.get_metadata("dtime")["dtime"]

    assert savepoint.nudgezone_diff() == diffusion_granule.nudgezone_diff
    assert savepoint.bdy_diff() == diffusion_granule.bdy_diff
    assert savepoint.fac_bdydiff_v() == diffusion_granule.fac_bdydiff_v
    assert savepoint.smag_offset() == diffusion_granule.smag_offset
    assert savepoint.diff_multfac_w() == diffusion_granule.diff_multfac_w

    # this is done in diffusion.run(...) because it depends on the dtime
    diffusion_utils.scale_k.with_backend(backend)(
        diffusion_granule.enh_smag_fac,
        dtime,
        diffusion_granule.diff_multfac_smag,
        offset_provider={},
    )
    assert test_utils.dallclose(
        diffusion_granule.enh_smag_fac.asnumpy(), savepoint.enh_smag_fac(), rtol=1e-7
    )
    assert test_utils.dallclose(
        diffusion_granule.diff_multfac_smag.asnumpy(), savepoint.diff_multfac_smag(), rtol=1e-7
    )

    assert test_utils.dallclose(diffusion_granule.smag_limit.asnumpy(), savepoint.smag_limit())
    assert test_utils.dallclose(
        diffusion_granule.diff_multfac_n2w.asnumpy(), savepoint.diff_multfac_n2w()
    )
    assert test_utils.dallclose(
        diffusion_granule.diff_multfac_vn.asnumpy(), savepoint.diff_multfac_vn()
    )


@pytest.mark.uses_concat_where
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description,step_date_init",
    [
        (definitions.Experiments.MCH_CH_R04B09, "2021-06-20T12:00:10.000"),
        (definitions.Experiments.MCH_CH_R04B09, "2021-06-20T12:00:20.000"),
        (definitions.Experiments.EXCLAIM_APE, "2000-01-01T00:00:02.000"),
        (definitions.Experiments.EXCLAIM_APE, "2000-01-01T00:00:04.000"),
    ],
)
def test_verify_diffusion_init_against_savepoint(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment,
    step_date_init,
    interpolation_state: diffusion_states.DiffusionInterpolationState,
    metric_state: diffusion_states.DiffusionMetricState,
    savepoint_diffusion_init,
    backend,
):
    grid = get_grid_for_experiment(experiment, backend)
    cell_params = get_cell_geometry_for_experiment(experiment, backend)
    edge_params = get_edge_geometry_for_experiment(experiment, backend)
    config = experiment.config.diffusion
    additional_parameters = diffusion.DiffusionParams(config)
    vertical_config = experiment.config.vertical_grid
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )

    diffusion_granule = diffusion.Diffusion(
        grid=grid,
        config=config,
        params=additional_parameters,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_params,
        cell_params=cell_params,
        backend=backend,
        exchange=decomp_defs.single_node_exchange,
    )

    _verify_init_values_against_savepoint(savepoint_diffusion_init, diffusion_granule, backend)


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
            definitions.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_run_diffusion_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment,
    step_date_init,
    step_date_exit,
    savepoint_diffusion_init,
    savepoint_diffusion_exit,
    interpolation_state: diffusion_states.DiffusionInterpolationState,
    metric_state: diffusion_states.DiffusionMetricState,
    backend,
):
    grid = get_grid_for_experiment(experiment, backend)
    cell_geometry = get_cell_geometry_for_experiment(experiment, backend)
    edge_geometry = get_edge_geometry_for_experiment(experiment, backend)

    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")

    diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )
    prognostic_state = savepoint_diffusion_init.construct_prognostics()

    vertical_config = experiment.config.vertical_grid
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )

    config = experiment.config.diffusion
    additional_parameters = diffusion.DiffusionParams(config)

    diffusion_granule = diffusion.Diffusion(
        grid=grid,
        config=config,
        params=additional_parameters,
        vertical_grid=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        backend=backend,
        exchange=decomp_defs.single_node_exchange,
    )
    verify_diffusion_fields(config, diagnostic_state, prognostic_state, savepoint_diffusion_init)
    assert savepoint_diffusion_init.fac_bdydiff_v() == diffusion_granule.fac_bdydiff_v

    diffusion_granule.run(
        diagnostic_state=diagnostic_state, prognostic_state=prognostic_state, dtime=dtime
    )
    verify_diffusion_fields(config, diagnostic_state, prognostic_state, savepoint_diffusion_exit)


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.MCH_CH_R04B09])
@pytest.mark.parametrize("linit", [True])
def test_run_diffusion_initial_step(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment,
    linit,
    savepoint_diffusion_init,
    savepoint_diffusion_exit,
    interpolation_state: diffusion_states.DiffusionInterpolationState,
    metric_state: diffusion_states.DiffusionMetricState,
    backend,
):
    grid = get_grid_for_experiment(experiment, backend)
    cell_geometry = get_cell_geometry_for_experiment(experiment, backend)
    edge_geometry = get_edge_geometry_for_experiment(experiment, backend)
    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")

    vertical_config = experiment.config.vertical_grid
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)
    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )
    diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )
    prognostic_state = savepoint_diffusion_init.construct_prognostics()
    config = experiment.config.diffusion
    params = diffusion.DiffusionParams(config)

    diffusion_granule = diffusion.Diffusion(
        grid=grid,
        config=config,
        params=params,
        vertical_grid=vertical_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        backend=backend,
        exchange=decomp_defs.single_node_exchange,
    )

    assert savepoint_diffusion_init.fac_bdydiff_v() == diffusion_granule.fac_bdydiff_v

    diffusion_granule.run(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        dtime=dtime,
        initial_run=True,
    )

    verify_diffusion_fields(
        config=config,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        diffusion_savepoint=savepoint_diffusion_exit,
    )


@pytest.mark.datatest
@pytest.mark.parametrize("linit", [True])
# TODO(havogt): Remove custom `experiment` parametrization
@pytest.mark.parametrize(
    "experiment_description,step_date_init",
    [
        (definitions.Experiments.MCH_CH_R04B09, "2021-06-20T12:00:10.000"),
    ],
)
def test_verify_special_diffusion_inital_step_values_against_initial_savepoint(
    savepoint_diffusion_init, experiment, icon_grid, linit, backend
):
    savepoint = savepoint_diffusion_init
    config = experiment.config.diffusion

    params = diffusion.DiffusionParams(config)
    expected_diff_multfac_vn = savepoint.diff_multfac_vn()
    expected_smag_limit = savepoint.smag_limit()
    exptected_smag_offset = savepoint.smag_offset()

    diff_multfac_vn = data_alloc.zero_field(icon_grid, dims.KDim, allocator=backend)
    smag_limit = data_alloc.zero_field(icon_grid, dims.KDim, allocator=backend)
    diffusion_utils.setup_fields_for_initial_step.with_backend(backend)(
        params.K4,
        config.hdiff_efdt_ratio,
        diff_multfac_vn,
        smag_limit,
        offset_provider={},
    )
    assert np.allclose(expected_smag_limit, smag_limit.asnumpy())
    assert np.allclose(expected_diff_multfac_vn, diff_multfac_vn.asnumpy())
    assert exptected_smag_offset == 0.0

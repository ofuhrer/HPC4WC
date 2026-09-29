# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Any

import pytest
from gt4py.next import typing as gtx_typing

from icon4py.model.atmosphere.diffusion import diffusion as diffusion_, diffusion_states
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import icon, vertical as v_grid
from icon4py.model.testing import definitions, parallel_helpers, serialbox

from .. import utils
from ..fixtures import *  # noqa: F403


_log = logging.getLogger(__file__)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.uses_concat_where
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (definitions.Experiments.EXCLAIM_APE, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
        (definitions.Experiments.JW, "2008-09-01T00:05:00.000", "2008-09-01T00:05:00.000"),
    ],
)
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_parallel_diffusion(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment: definitions.Experiment,
    step_date_init: str,
    step_date_exit: str,
    linit: bool,
    process_props: decomp_defs.ProcessProperties,
    decomposition_info: decomp_defs.DecompositionInfo,
    icon_grid: icon.IconGrid,
    savepoint_diffusion_init: serialbox.IconDiffusionInitSavepoint,
    savepoint_diffusion_exit: serialbox.IconDiffusionExitSavepoint,
    grid_savepoint: serialbox.IconGridSavepoint,
    metric_state: diffusion_states.DiffusionMetricState,
    interpolation_state: diffusion_states.DiffusionInterpolationState,
    caplog: Any,
    backend: gtx_typing.Backend,
) -> None:
    caplog.set_level("INFO")
    parallel_helpers.check_comm_size(process_props)
    _log.info(
        f"rank={process_props.rank}/{process_props.comm_size}: initializing diffusion for experiment '{definitions.Experiments.MCH_CH_R04B09}'"
    )
    _log.info(
        f"local cells = {decomposition_info.global_index(dims.CellDim, decomp_defs.DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomposition_info.global_index(dims.EdgeDim, decomp_defs.DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomposition_info.global_index(dims.VertexDim, decomp_defs.DecompositionInfo.EntryType.ALL).shape}"
    )
    _log.info(
        f"rank={process_props.rank}/{process_props.comm_size}:  GHEX context setup: from {process_props.comm_name} with {process_props.comm_size} nodes"
    )

    _log.info(
        f"rank={process_props.rank}/{process_props.comm_size}: using local grid with {icon_grid.num_cells} Cells, {icon_grid.num_edges} Edges, {icon_grid.num_vertices} Vertices"
    )
    config = experiment.config.diffusion
    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")
    _log.info(
        f"rank={process_props.rank}/{process_props.comm_size}:  setup: using {process_props.comm_name} with {process_props.comm_size} nodes"
    )
    vertical_config = experiment.config.vertical_grid

    diffusion_params = diffusion_.DiffusionParams(config)
    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()
    exchange = decomp_defs.create_exchange(process_props, decomposition_info)
    diffusion = diffusion_.Diffusion(
        grid=icon_grid,
        config=config,
        params=diffusion_params,
        vertical_grid=v_grid.VerticalGrid(
            vertical_config,
            grid_savepoint.vct_a(),
            grid_savepoint.vct_b(),
        ),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        exchange=exchange,
        backend=backend,
    )

    _log.info(f"rank={process_props.rank}/{process_props.comm_size}: diffusion initialized ")

    diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )

    prognostic_state = savepoint_diffusion_init.construct_prognostics()
    diffusion.run(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        dtime=dtime,
        initial_run=linit,
    )
    _log.info(f"rank={process_props.rank}/{process_props.comm_size}: diffusion run ")

    utils.verify_diffusion_fields(
        config=config,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        diffusion_savepoint=savepoint_diffusion_exit,
    )
    _log.info(
        f"rank={process_props.rank}/{process_props.comm_size}:  running diffusion step - using {process_props.comm_name} with {process_props.comm_size} nodes - DONE"
    )

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver.testcases import jablonowski_williamson as jabw
from icon4py.model.testing import datatest_utils as dt_utils, definitions, test_utils
from icon4py.model.testing.fixtures.datatest import backend

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import pathlib

    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description, rank", [(definitions.Experiments.JW, 0)])
def test_jabw_initial_condition(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment: definitions.Experiment,
    process_props: decomposition.ProcessProperties,
    backend: gtx_typing.Backend,
    rank: int,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    icon_grid: base_grid.Grid,
):
    edge_geometry = grid_savepoint.construct_edge_geometry()
    cell_geometry = grid_savepoint.construct_cell_geometry()
    default_surface_pressure = data_alloc.constant_field(icon_grid, 1e5, dims.CellDim)

    (
        _,
        solve_nonhydro_diagnostic_state,
        _,
        _,
        diagnostic_state,
        prognostic_state_now,
        _,
    ) = jabw.model_initialization_jabw(
        grid=icon_grid,
        cell_param=cell_geometry,
        edge_param=edge_geometry,
        path=dt_utils.get_datapath_for_experiment(experiment.description, process_props),
        backend=backend,
        rank=rank,
    )

    # note that w is not verified because we decided to force w to zero in python framework after discussion
    jabw_exit_savepoint = data_provider.from_savepoint_jabw_exit()
    assert test_utils.dallclose(
        prognostic_state_now.rho.asnumpy(),
        jabw_exit_savepoint.rho().asnumpy(),
    )

    assert test_utils.dallclose(
        prognostic_state_now.exner.asnumpy(),
        jabw_exit_savepoint.exner().asnumpy(),
    )

    assert test_utils.dallclose(
        prognostic_state_now.theta_v.asnumpy(),
        jabw_exit_savepoint.theta_v().asnumpy(),
    )

    assert test_utils.dallclose(
        prognostic_state_now.vn.asnumpy(),
        jabw_exit_savepoint.vn().asnumpy(),
    )

    assert test_utils.dallclose(
        diagnostic_state.pressure.asnumpy(),
        jabw_exit_savepoint.pressure().asnumpy(),
    )

    assert test_utils.dallclose(
        diagnostic_state.temperature.asnumpy(),
        jabw_exit_savepoint.temperature().asnumpy(),
    )

    assert test_utils.dallclose(
        diagnostic_state.surface_pressure.asnumpy(),
        default_surface_pressure.asnumpy(),
    )

    assert test_utils.dallclose(
        solve_nonhydro_diagnostic_state.perturbed_exner_at_cells_on_model_levels.asnumpy(),
        data_provider.from_savepoint_diagnostics_initial().exner_pr().asnumpy(),
        atol=1.0e-14,
    )

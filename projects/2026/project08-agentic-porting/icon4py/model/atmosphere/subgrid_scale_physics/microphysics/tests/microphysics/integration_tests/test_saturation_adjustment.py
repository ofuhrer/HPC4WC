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

from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    saturation_adjustment as satad,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.embedded_static_args
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description",
    [definitions.Experiments.WEISMAN_KLEMP_TORUS],
)
@pytest.mark.parametrize(
    "date", ["2008-09-01T01:59:48.000", "2008-09-01T01:59:52.000", "2008-09-01T01:59:56.000"]
)
@pytest.mark.parametrize("location", ["nwp-gscp-interface", "interface-nwp"])
def test_saturation_adjustement(
    location: str,
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    icon_grid: base_grid.Grid,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    satad_init = data_provider.from_savepoint_satad_init(location=location, date=date)
    satad_exit = data_provider.from_savepoint_satad_exit(location=location, date=date)
    entry_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_entry(date=date)

    config = satad.SaturationAdjustmentConfig()
    dtime = entry_savepoint.dtime()

    vertical_config = experiment.config.vertical_grid
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
    )
    temperature_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, allocator=backend
    )
    qv_tendency = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    qc_tendency = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)

    metric_state = satad.MetricStateSaturationAdjustment(
        ddqz_z_full=metrics_savepoint.ddqz_z_full()
    )

    saturation_adjustment = satad.SaturationAdjustment(
        config=config,
        grid=icon_grid,
        metric_state=metric_state,
        vertical_params=vertical_params,
        backend=backend,
    )

    qv = satad_init.qv()
    qc = satad_init.qc()
    rho = satad_init.rho()
    temperature = satad_init.temperature()

    # run saturation adjustment
    saturation_adjustment.run(
        dtime=dtime,
        rho=rho,
        temperature=temperature,
        qv=qv,
        qc=qc,
        temperature_tendency=temperature_tendency,
        qv_tendency=qv_tendency,
        qc_tendency=qc_tendency,
    )

    updated_qv = qv.asnumpy() + qv_tendency.asnumpy() * dtime
    updated_qc = qc.asnumpy() + qc_tendency.asnumpy() * dtime
    updated_temperature = temperature.asnumpy() + temperature_tendency.asnumpy() * dtime

    assert test_utils.dallclose(
        updated_qv,
        satad_exit.qv().asnumpy(),
        atol=1.0e-13,
    )
    assert test_utils.dallclose(
        updated_qc,
        satad_exit.qc().asnumpy(),
        atol=1.0e-13,
    )
    assert test_utils.dallclose(
        updated_temperature,
        satad_exit.temperature().asnumpy(),
        atol=1.0e-13,
    )

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
    microphysics_options as mphys_options,
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    tracer_state as tracers,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid
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
def test_graupel(
    experiment: definitions.Experiment,
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    icon_grid: icon_grid.IconGrid,
    backend: gtx_typing.Backend,
):
    vertical_config = experiment.config.vertical_grid
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
    )

    metric_state = graupel.MetricStateIconGraupel(
        ddqz_z_full=metrics_savepoint.ddqz_z_full(),
    )

    entry_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_entry(date=date)
    exit_savepoint = data_provider.from_savepoint_weisman_klemp_graupel_exit(date=date)

    dtime = entry_savepoint.dtime()

    tracer_state = tracers.TracerState(
        qv=entry_savepoint.qv(),
        qc=entry_savepoint.qc(),
        qr=entry_savepoint.qr(),
        qi=entry_savepoint.qi(),
        qs=entry_savepoint.qs(),
        qg=entry_savepoint.qg(),
    )
    prognostic_state = prognostics.PrognosticState(
        rho=entry_savepoint.rho(), vn=None, w=None, exner=None, theta_v=None
    )
    diagnostic_state = diagnostics.DiagnosticState(
        temperature=entry_savepoint.temperature(),
        virtual_temperature=None,
        pressure=entry_savepoint.pressure(),
        pressure_ifc=None,
        u=None,
        v=None,
    )

    graupel_config = experiment.config.graupel
    assert graupel_config is not None, "expected microphysics configuration for this experiment"

    graupel_microphysics = graupel.SingleMomentSixClassIconGraupel(
        graupel_config=graupel_config,
        grid=icon_grid,
        metric_state=metric_state,
        vertical_params=vertical_params,
        backend=backend,
    )

    qnc = entry_savepoint.qnc()

    temperature_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    qv_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    qc_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    qr_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    qi_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    qs_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    qg_tendency = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )

    graupel_microphysics.run(
        dtime=dtime,
        rho=prognostic_state.rho,
        temperature=diagnostic_state.temperature,
        pressure=diagnostic_state.pressure,
        qv=tracer_state.qv,
        qc=tracer_state.qc,
        qr=tracer_state.qr,
        qi=tracer_state.qi,
        qs=tracer_state.qs,
        qg=tracer_state.qg,
        qnc=qnc,
        temperature_tendency=temperature_tendency,
        qv_tendency=qv_tendency,
        qc_tendency=qc_tendency,
        qr_tendency=qr_tendency,
        qi_tendency=qi_tendency,
        qs_tendency=qs_tendency,
        qg_tendency=qg_tendency,
    )

    new_temperature = (
        entry_savepoint.temperature().asnumpy() + temperature_tendency.asnumpy() * dtime
    )
    new_qv = entry_savepoint.qv().asnumpy() + qv_tendency.asnumpy() * dtime
    new_qc = entry_savepoint.qc().asnumpy() + qc_tendency.asnumpy() * dtime
    new_qr = entry_savepoint.qr().asnumpy() + qr_tendency.asnumpy() * dtime
    new_qi = entry_savepoint.qi().asnumpy() + qi_tendency.asnumpy() * dtime
    new_qs = entry_savepoint.qs().asnumpy() + qs_tendency.asnumpy() * dtime
    new_qg = entry_savepoint.qg().asnumpy() + qg_tendency.asnumpy() * dtime

    assert test_utils.dallclose(
        new_temperature,
        exit_savepoint.temperature().asnumpy(),
    )
    assert test_utils.dallclose(
        new_qv,
        exit_savepoint.qv().asnumpy(),
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qc,
        exit_savepoint.qc().asnumpy(),
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qr,
        exit_savepoint.qr().asnumpy(),
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qi,
        exit_savepoint.qi().asnumpy(),
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qs,
        exit_savepoint.qs().asnumpy(),
        atol=1.0e-12,
    )
    assert test_utils.dallclose(
        new_qg,
        exit_savepoint.qg().asnumpy(),
        atol=1.0e-12,
    )

    assert test_utils.dallclose(
        graupel_microphysics.rain_precipitation_flux.asnumpy()[:, -1],
        exit_savepoint.rain_flux().asnumpy()[:],
        atol=9.0e-11,
    )
    assert test_utils.dallclose(
        graupel_microphysics.snow_precipitation_flux.asnumpy()[:, -1],
        exit_savepoint.snow_flux().asnumpy()[:],
        atol=9.0e-11,
    )
    assert test_utils.dallclose(
        graupel_microphysics.graupel_precipitation_flux.asnumpy()[:, -1],
        exit_savepoint.graupel_flux().asnumpy()[:],
        atol=9.0e-11,
    )
    assert test_utils.dallclose(
        graupel_microphysics.ice_precipitation_flux.asnumpy()[:, -1],
        exit_savepoint.ice_flux().asnumpy()[:],
        atol=9.0e-11,
    )

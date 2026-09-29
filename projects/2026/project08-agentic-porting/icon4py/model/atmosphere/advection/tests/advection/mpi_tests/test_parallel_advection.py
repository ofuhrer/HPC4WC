# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from gt4py.next import typing as gtx_typing

import icon4py.model.testing.test_utils as test_helpers
from icon4py.model.atmosphere.advection import advection
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import (
    base as base_grid,
    geometry_attributes as geometry_attrs,
    horizontal as h_grid,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    definitions as test_defs,
    grid_utils,
    parallel_helpers,
    serialbox as sb,
    test_utils,
)
from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    process_props,
)

from ..fixtures import *  # noqa: F403
from ..utils import (
    construct_diagnostic_exit_state,
    construct_diagnostic_init_state,
    construct_interpolation_state,
    construct_least_squares_state,
    construct_metric_state,
    construct_prep_adv,
    log_serialized,
)


@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.MCH_CH_R04B09])
@pytest.mark.parametrize(
    "date, even_timestep, ntracer, horizontal_advection_type, horizontal_advection_limiter, vertical_advection_type, vertical_advection_limiter",
    [
        (
            "2021-06-20T12:00:10.000",
            False,
            1,
            advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
            advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
            advection.VerticalAdvectionType.NO_ADVECTION,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
        ),
        (
            "2021-06-20T12:00:20.000",
            True,
            1,
            advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
            advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
            advection.VerticalAdvectionType.NO_ADVECTION,
            advection.VerticalAdvectionLimiter.NO_LIMITER,
        ),
        (
            "2021-06-20T12:00:10.000",
            False,
            4,
            advection.HorizontalAdvectionType.NO_ADVECTION,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.PPM_3RD_ORDER,
            advection.VerticalAdvectionLimiter.SEMI_MONOTONIC,
        ),
        (
            "2021-06-20T12:00:20.000",
            True,
            4,
            advection.HorizontalAdvectionType.NO_ADVECTION,
            advection.HorizontalAdvectionLimiter.NO_LIMITER,
            advection.VerticalAdvectionType.PPM_3RD_ORDER,
            advection.VerticalAdvectionLimiter.SEMI_MONOTONIC,
        ),
    ],
)
@pytest.mark.mpi
def test_advection_run_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    date,
    even_timestep,
    ntracer,
    horizontal_advection_type,
    horizontal_advection_limiter,
    vertical_advection_type,
    vertical_advection_limiter,
    *,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    backend,
    advection_init_savepoint,
    advection_exit_savepoint,
    experiment: test_defs.Experiment,
    process_props: definitions.ProcessProperties,
    decomposition_info: definitions.DecompositionInfo,
    construct_advection_lsq_state,
):
    if test_utils.is_embedded(backend):
        # https://github.com/GridTools/gt4py/issues/1583
        pytest.xfail("ValueError: axes don't match array")
    # TODO(OngChia): the last datatest fails on GPU (or even CPU) backend when there is no advection because the horizontal flux is not zero. Further check required.
    if (
        even_timestep
        and horizontal_advection_type == advection.HorizontalAdvectionType.NO_ADVECTION
    ):
        pytest.xfail(
            "This test is skipped until the cause of nonzero horizontal advection if revealed."
        )

    parallel_helpers.check_comm_size(process_props)
    parallel_helpers.log_process_properties(process_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    config = advection.AdvectionConfig(
        horizontal_advection_type=horizontal_advection_type,
        horizontal_advection_limiter=horizontal_advection_limiter,
        vertical_advection_type=vertical_advection_type,
        vertical_advection_limiter=vertical_advection_limiter,
    )

    interpolation_state = construct_interpolation_state(
        savepoint=interpolation_savepoint, backend=backend
    )

    metric_state = construct_metric_state(
        icon_grid=icon_grid, savepoint=metrics_savepoint, backend=backend
    )
    edge_geometry = grid_savepoint.construct_edge_geometry()
    cell_geometry = grid_savepoint.construct_cell_geometry()
    exchange_runtime = definitions.create_exchange(process_props, decomposition_info)

    advection_granule = advection.convert_config_to_advection(
        config=config,
        grid=icon_grid,
        interpolation_state=interpolation_state,
        least_squares_state=construct_advection_lsq_state,
        metric_state=metric_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        even_timestep=even_timestep,
        backend=backend,
        exchange=exchange_runtime,
    )

    diagnostic_state = construct_diagnostic_init_state(
        icon_grid=icon_grid, savepoint=advection_init_savepoint, ntracer=ntracer, backend=backend
    )
    prep_adv = construct_prep_adv(advection_init_savepoint)
    p_tracer_now = advection_init_savepoint.tracer(ntracer)

    p_tracer_new = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    dtime = advection_init_savepoint.get_metadata("dtime").get("dtime")

    log_serialized(diagnostic_state, prep_adv, p_tracer_now, dtime)

    advection_granule.run(
        diagnostic_state=diagnostic_state,
        prep_adv=prep_adv,
        p_tracer_now=p_tracer_now,
        p_tracer_new=p_tracer_new,
        dtime=dtime,
    )

    diagnostic_state_ref = construct_diagnostic_exit_state(
        icon_grid=icon_grid,
        savepoint=advection_exit_savepoint,
        ntracer=ntracer,
        backend=backend,
    )
    p_tracer_new_ref = advection_exit_savepoint.tracer(ntracer)

    test_helpers.assert_dallclose(
        diagnostic_state.hfl_tracer.asnumpy(),
        diagnostic_state_ref.hfl_tracer.asnumpy(),
        atol=1e-11,
    )

    test_utils.assert_dallclose(
        diagnostic_state.vfl_tracer.asnumpy(),
        diagnostic_state_ref.vfl_tracer.asnumpy(),
        rtol=1e-10,
    )

    test_helpers.assert_dallclose(
        p_tracer_new_ref.asnumpy(),
        p_tracer_new.asnumpy(),
        atol=1e-16,
    )

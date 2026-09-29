# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.testing.fixtures import (
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    istep_exit,
    istep_init,
    metrics_savepoint,
    process_props,
    savepoint_diffusion_exit,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_init,
    savepoint_nonhydro_step_final,
    savepoint_velocity_init,
    step_date_exit,
    step_date_init,
)


@pytest.fixture
def timeloop_diffusion_savepoint_init(
    data_provider,  # imported fixtures data_provider
    step_date_init,  # imported fixtures data_provider
    timeloop_diffusion_linit_init,
):
    """
    Load data from ICON savepoint at start of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'timeloop_date_'
    fixture, passing 'step_date_init=<iso_string>'

    linit flag is set to true
    """
    return data_provider.from_savepoint_diffusion_init(
        linit=timeloop_diffusion_linit_init, date=step_date_init
    )


@pytest.fixture
def linit(timeloop_diffusion_linit_exit: bool) -> bool:
    return timeloop_diffusion_linit_exit


@pytest.fixture
def timeloop_date_init():
    return "2021-06-20T12:00:00.000"


@pytest.fixture
def timeloop_date_exit():
    return "2021-06-20T12:00:10.000"

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

from icon4py.model.testing.fixtures import (
    backend,
    backend_like,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
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
    topography_savepoint,
)


@pytest.fixture
def linit(timeloop_diffusion_linit_exit: bool) -> bool:
    return timeloop_diffusion_linit_exit

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.testing.fixtures.datatest import (
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    istep_exit,
    istep_init,
    linit,
    metrics_savepoint,
    process_props,
    savepoint_diffusion_exit,
    savepoint_diffusion_init,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_init,
    savepoint_nonhydro_step_final,
    step_date_exit,
    step_date_init,
    substep_exit,
    substep_init,
)


# Make sure custom icon4py pytest hooks are loaded
try:
    import sys

    _ = sys.modules["icon4py.model.testing.pytest_config"]
except KeyError:
    from icon4py.model.testing.pytest_hooks import *  # noqa: F403


__all__ = [
    # imported fixtures:
    "data_provider",
    "download_ser_data",
    "experiment",
    "experiment_description",
    "grid_savepoint",
    "icon_grid",
    "interpolation_savepoint",
    "istep_exit",
    "istep_init",
    "linit",
    "metrics_savepoint",
    "process_props",
    "savepoint_diffusion_exit",
    "savepoint_diffusion_init",
    "savepoint_nonhydro_exit",
    "savepoint_nonhydro_init",
    "savepoint_nonhydro_step_final",
    "step_date_exit",
    "step_date_init",
    "substep_exit",
    "substep_init",
]

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from icon4py.model.atmosphere.advection import advection_states
from icon4py.model.testing import serialbox
from icon4py.model.testing.fixtures.datatest import data_provider, decomposition_info


@pytest.fixture
def date() -> str:
    """
    Fixture to provide a date (as a iso_string) for loading ICON savepoints.

    This is just a placeholder. It MUST be overridden by a parametrized
    argument with the same name and a relevant value in the test function.
    """
    return "0000-00-00T00:00:00Z"


@pytest.fixture
def advection_init_savepoint(data_provider, date):
    """
    Load data from advection init ICON savepoint.

    Date of the timestamp to be selected MUST be set separately by overriding the 'date'
    fixture, passing 'date=<iso_string>'.
    """
    return data_provider.from_advection_init_savepoint(size=data_provider.grid_size, date=date)


@pytest.fixture
def advection_exit_savepoint(data_provider, date):
    """
    Load data from advection exit ICON savepoint.

    Date of the timestamp to be selected MUST be set separately by overriding the 'date'
    fixture, passing 'date=<iso_string>'.
    """
    return data_provider.from_advection_exit_savepoint(size=data_provider.grid_size, date=date)


@pytest.fixture
def construct_advection_lsq_state(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
) -> advection_states.AdvectionLeastSquaresState:
    return advection_states.AdvectionLeastSquaresState(
        lsq_pseudoinv_1=interpolation_savepoint.lsq_pseudoinv_1(),
        lsq_pseudoinv_2=interpolation_savepoint.lsq_pseudoinv_2(),
    )

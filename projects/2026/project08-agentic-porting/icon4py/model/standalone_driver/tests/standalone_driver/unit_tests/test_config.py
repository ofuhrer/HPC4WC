# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``standalone_driver.config`` (data-free)."""

import datetime

import pytest

from icon4py.model.standalone_driver import config as driver_config


def _make_dicts(run_nml: dict) -> tuple[dict, dict]:
    atm_dict = {
        "nonhydrostatic_nml": {"vcfl_threshold": 0.85, "ndyn_substeps": 5},
        "run_nml": run_nml,
    }
    master_dict = {
        "master_time_control_nml": {
            "experimentstartdate": "2000-01-01T00:00:00Z",
            "experimentstopdate": "2000-01-01T01:00:00Z",
        },
        "master_model_nml": {"model_namelist_filename": "NAMELIST_test_sb_atm"},
    }
    return atm_dict, master_dict


@pytest.mark.parametrize(
    ("duration", "expected_seconds"),
    [
        ("PT300S", 300.0),
        ("PT1H", 3600.0),
        ("PT10M", 600.0),
        ("PT1H30M", 5400.0),
        ("P1DT6H", 108000.0),
        ("PT0.5S", 0.5),
    ],
)
def test_timedelta_from_iso8601_valid(duration: str, expected_seconds: float) -> None:
    assert driver_config._timedelta_from_iso8601(duration) == datetime.timedelta(
        seconds=expected_seconds
    )


@pytest.mark.parametrize("duration", ["", "P", "PT", "P1Y", "P1M", "300", "PT300", "P1DT", "P1WT"])
def test_timedelta_from_iso8601_invalid(duration: str) -> None:
    with pytest.raises(ValueError, match="Invalid ISO 8601 duration"):
        driver_config._timedelta_from_iso8601(duration)


def test_modeltimestep_takes_priority_over_dtime() -> None:
    # trailing whitespace mimics the fixed-width Fortran string
    atm_dict, master_dict = _make_dicts(
        {"dtime": 999.0, "modeltimestep": "PT300S                          "}
    )
    config = driver_config.DriverConfig.from_fortran_dict(
        atm_dict=atm_dict, master_dict=master_dict, profiling_stats=None
    )
    assert config.dtime == datetime.timedelta(seconds=300)


def test_empty_modeltimestep_falls_back_to_dtime() -> None:
    atm_dict, master_dict = _make_dicts({"dtime": 120.0, "modeltimestep": "        "})
    config = driver_config.DriverConfig.from_fortran_dict(
        atm_dict=atm_dict, master_dict=master_dict, profiling_stats=None
    )
    assert config.dtime == datetime.timedelta(seconds=120)

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import enum
import pathlib

import pytest

from icon4py.model.testing import config, data_handling, datatest_utils as dt_utils, definitions


def _path_to_experiment_testdata(experiment: MuphysExperiment) -> pathlib.Path:
    return (
        config.TEST_DATA_PATH
        / definitions.MUPHYS_DATA_DIR
        / experiment.type.name.lower()
        / experiment.name
    )


class ExperimentType(int, enum.Enum):
    GRAUPEL_ONLY = 0
    FULL_MUPHYS = 1


@dataclasses.dataclass(frozen=True)
class MuphysExperiment:
    name: str
    type: ExperimentType
    dt: float = 30.0
    qnc: float = 100.0

    @property
    def input_file(self) -> pathlib.Path:
        return _path_to_experiment_testdata(self) / "input.nc"

    @property
    def reference_file(self) -> pathlib.Path:
        return _path_to_experiment_testdata(self) / "reference.nc"

    def __str__(self):
        return self.name


@pytest.fixture(autouse=True)
def download_test_data(experiment: MuphysExperiment) -> None:
    """Downloads test data for an experiment (implicit fixture)."""
    data_handling.download_test_data(
        _path_to_experiment_testdata(experiment),
        uri=dt_utils.get_muphys_archive_url(
            definitions.TESTDATA_ROOT_URL,
            experiment.type.name.lower(),
            experiment.name,
        ),
    )

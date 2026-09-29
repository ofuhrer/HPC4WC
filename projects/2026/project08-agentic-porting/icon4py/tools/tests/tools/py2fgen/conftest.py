# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import os
import pathlib
import sys

import pytest


@pytest.fixture
def samples_path():
    return pathlib.Path(__file__).parent / "fortran_samples"


@pytest.fixture
def fortran_subprocess_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward parent sys.path through PYTHONPATH for the test's duration.

    A subprocess that embeds Python via CFFI inherits the env's PYTHONPATH
    but not pytest's rootdir-induced sys.path additions; without this, the
    embedded interpreter cannot resolve pytest-discovered packages.
    """
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, [*sys.path, os.environ.get("PYTHONPATH", "")])),
    )

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import difflib
import pathlib

import pytest
from click.testing import CliRunner

from icon4py.bindings import all_bindings
from icon4py.tools.py2fgen import _utils


logger = _utils.setup_logger(__name__)

# The .c source is not snapshotted: it is bulky CFFI-generated code that
# varies between cffi versions.
_SNAPSHOT_SUFFIXES = (".py", ".f90", ".h")


@pytest.fixture
def cli_runner():
    return CliRunner()


def _reference(suffix: str) -> pathlib.Path:
    base = pathlib.Path(__file__).parent.resolve() / "references"
    return base / f"{all_bindings.LIBRARY_NAME}{suffix}"


def _actual(suffix: str) -> pathlib.Path:
    base = pathlib.Path(__file__).parent.resolve() / "references_new"
    return base / f"{all_bindings.LIBRARY_NAME}{suffix}"


def diff(reference: pathlib.Path, actual: pathlib.Path) -> bool:
    with pathlib.Path.open(reference) as f:
        reference_lines = f.readlines()
    with pathlib.Path.open(actual) as f:
        actual_lines = f.readlines()

    clean = True
    for line in difflib.context_diff(reference_lines, actual_lines):
        logger.info(f"result line: {line}")
        clean = False
    return clean


def test_references(cli_runner):
    cli_args = []
    for suffix in _SNAPSHOT_SUFFIXES:
        cli_args += [f"--output-{suffix[1:]}", str(_actual(suffix))]
    result = cli_runner.invoke(all_bindings.main, cli_args)
    assert result.exit_code == 0, result.output

    for suffix in _SNAPSHOT_SUFFIXES:
        assert diff(_reference(suffix), _actual(suffix))

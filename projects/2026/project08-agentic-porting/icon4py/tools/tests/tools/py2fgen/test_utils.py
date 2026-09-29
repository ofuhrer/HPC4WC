# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

from icon4py.tools.py2fgen._utils import write_if_changed


def test_write_if_changed_creates_new_file(tmp_path: pathlib.Path):
    path = tmp_path / "test.txt"
    assert write_if_changed("hello", path) is True
    assert path.read_text() == "hello"


def test_write_if_changed_skips_unchanged(tmp_path: pathlib.Path):
    path = tmp_path / "test.txt"
    write_if_changed("hello", path)
    assert write_if_changed("hello", path) is False
    assert path.read_text() == "hello"


def test_write_if_changed_writes_when_changed(tmp_path: pathlib.Path):
    path = tmp_path / "test.txt"
    write_if_changed("hello", path)
    assert write_if_changed("world", path) is True
    assert path.read_text() == "world"


def test_write_if_changed_force_rewrites_unchanged(tmp_path: pathlib.Path):
    path = tmp_path / "test.txt"
    write_if_changed("hello", path)
    assert write_if_changed("hello", path, force=True) is True
    assert path.read_text() == "hello"


def test_write_if_changed_creates_parent_dirs(tmp_path: pathlib.Path):
    path = tmp_path / "nested" / "dir" / "test.txt"
    assert write_if_changed("hello", path) is True
    assert path.read_text() == "hello"

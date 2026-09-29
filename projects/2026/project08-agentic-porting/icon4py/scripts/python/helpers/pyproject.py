# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

"""Helpers for reading pyproject.toml files and uv workspace metadata."""

from __future__ import annotations

import pathlib
import tomllib


def read_pyproject(project_root: pathlib.Path) -> dict:
    with pathlib.Path.open(project_root / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def get_workspace_members(pyproject: dict) -> list[str]:
    return pyproject.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])


def get_package_name(project_dir: pathlib.Path) -> str:
    """Read the package name from a directory's pyproject.toml."""
    data = read_pyproject(project_dir)
    return data.get("project", {}).get("name", project_dir.name)

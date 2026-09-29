#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Discover icon4py workspace packages and output them in dependency order.

Reads the uv workspace members from the root pyproject.toml, resolves the
PyPI package name and directory for each member (plus the root metapackage),
computes a topological ordering based on workspace-internal dependencies,
and outputs a JSON array suitable for use as a GitHub Actions matrix.
"""

from __future__ import annotations

import graphlib
import json
import pathlib
import sys
from typing import Annotated

import packaging.requirements
import typer
from helpers import common, pyproject


cli = typer.Typer(no_args_is_help=True, help=__doc__)


def _get_all_dependencies(pyproject_data: dict) -> list[str]:
    deps = list(pyproject_data.get("project", {}).get("dependencies", []))
    optional = pyproject_data.get("project", {}).get("optional-dependencies", {})
    for group_deps in optional.values():
        deps.extend(group_deps)
    return deps


def _extract_workspace_dep_names(
    deps: list[str], known_names: set[str], self_name: str
) -> set[str]:
    workspace_deps = set()
    for dep in deps:
        req = packaging.requirements.Requirement(dep)
        if req.name in known_names and req.name != self_name:
            workspace_deps.add(req.name)
    return workspace_deps


def _discover(repo_root: pathlib.Path | None = None) -> list[dict[str, str]]:
    """Discover all workspace packages in dependency order.

    Returns a list of dicts with 'name' (PyPI package name) and 'dir'
    (relative path from repo root) for each package, ordered so that
    dependencies come before dependents.
    """
    root = repo_root or common.REPO_ROOT
    root_pyproject = pyproject.read_pyproject(root)

    members = pyproject.get_workspace_members(root_pyproject)
    root_name = root_pyproject.get("project", {}).get("name", "")

    packages: dict[str, dict[str, str]] = {}
    for member_dir in members:
        member_path = root / member_dir
        if not (member_path / "pyproject.toml").exists():
            continue
        name = pyproject.get_package_name(member_path)
        if name:
            packages[name] = {"name": name, "dir": member_dir}

    if root_name:
        packages[root_name] = {"name": root_name, "dir": "."}

    known_names = set(packages.keys())
    dep_graph: dict[str, set[str]] = {}
    for name, info in packages.items():
        pkg_path = root / info["dir"]
        pkg_pyproject = pyproject.read_pyproject(pkg_path)
        deps = _get_all_dependencies(pkg_pyproject)
        dep_graph[name] = _extract_workspace_dep_names(deps, known_names, name)

    ordered_names = list(graphlib.TopologicalSorter(dep_graph).static_order())
    return [packages[name] for name in ordered_names]


@cli.command()
def discover_packages(
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: 'json' or 'names'."),
    ] = "json",
    repo_root: Annotated[
        pathlib.Path | None,
        typer.Option("--repo-root", help="Repository root directory."),
    ] = None,
) -> None:
    """List all icon4py workspace packages in dependency order."""
    root = repo_root or common.REPO_ROOT
    packages = _discover(root)

    if output_format == "json":
        typer.echo(json.dumps(packages))
    elif output_format == "names":
        for pkg in packages:
            typer.echo(pkg["name"])
    else:
        typer.echo(f"Unknown format: {output_format}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    sys.exit(cli())

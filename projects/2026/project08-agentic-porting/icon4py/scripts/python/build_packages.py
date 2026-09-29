#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --only-group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Build Python distribution packages for icon4py and all its uv workspace members."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import textwrap
from typing import Annotated

import typer
from helpers import common, pyproject


cli = typer.Typer(
    name=__name__.split(".")[-1].replace("_", "-"), no_args_is_help=True, help=__doc__
)


def _run_uv_build(
    source_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    sdist: bool = True,
    wheel: bool = True,
    verbose: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    cmd = ["uv", "build", str(source_dir), "--out-dir", str(output_dir)]
    if sdist and not wheel:
        cmd.append("--sdist")
    elif wheel and not sdist:
        cmd.append("--wheel")
    if verbose:
        cmd.append("--verbose")
    return subprocess.run(cmd, capture_output=not verbose, check=False)


def _collect_targets(
    packages: list[str] | None,
) -> list[tuple[pathlib.Path, str]]:
    """Resolve build targets from optional member paths."""
    root_pyproject = pyproject.read_pyproject(common.REPO_ROOT)
    workspace_members = pyproject.get_workspace_members(root_pyproject)

    if packages:
        member_paths: list[str] = []
        for pkg in packages:
            if pkg in workspace_members or (common.REPO_ROOT / pkg / "pyproject.toml").is_file():
                member_paths.append(pkg)
            else:
                typer.echo(f"Warning: '{pkg}' is not a valid workspace member, skipping.", err=True)
    else:
        member_paths = list(workspace_members)

    targets: list[tuple[pathlib.Path, str]] = []
    seen_paths: set[pathlib.Path] = set()
    for member_path in member_paths:
        target_path = common.REPO_ROOT / member_path
        normalized_path = target_path.resolve()
        if normalized_path in seen_paths:
            continue
        seen_paths.add(normalized_path)
        targets.append((target_path, pyproject.get_package_name(target_path)))

    root_name = root_pyproject.get("project", {}).get("name", "icon4py")
    root_path = common.REPO_ROOT
    normalized_root_path = root_path.resolve()
    if normalized_root_path not in seen_paths:
        targets.append((root_path, root_name))

    return targets


@cli.command()
def build(
    output_dir: Annotated[
        pathlib.Path,
        typer.Option("--output-dir", "-o", help="Output directory for built distributions."),
    ] = pathlib.Path("dist"),
    sdist: Annotated[bool, typer.Option(help="Build source distributions.")] = True,
    wheel: Annotated[bool, typer.Option(help="Build wheel distributions.")] = True,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose output.")
    ] = False,
    packages: Annotated[
        list[str] | None,
        typer.Argument(
            help=(
                "Workspace member paths to build (e.g. 'model/common tools'). "
                "If omitted, all workspace members and the root package are built."
            ),
        ),
    ] = None,
) -> None:
    """Build Python distribution packages for icon4py and all workspace members."""
    if not sdist and not wheel:
        typer.echo("Error: at least one of --sdist or --wheel must be enabled.", err=True)
        raise typer.Exit(code=1)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_targets = _collect_targets(packages)

    succeeded = 0
    failed: list[str] = []

    for source_dir, name in build_targets:
        typer.echo(f"Building {name} ({source_dir.relative_to(common.REPO_ROOT) or '.'}) ...")
        result = _run_uv_build(source_dir, output_dir, sdist=sdist, wheel=wheel, verbose=verbose)
        if result.returncode == 0:
            succeeded += 1
            typer.echo(f"  OK  {name}")
        else:
            failed.append(name)
            typer.echo(f"  FAIL  {name}", err=True)
            if not verbose and result.stderr:
                sys.stderr.buffer.write(result.stderr)

    typer.echo(f"\n{succeeded}/{len(build_targets)} packages built -> {output_dir}")
    if failed:
        typer.echo(f"Failed: {', '.join(failed)}", err=True)
        raise typer.Exit(code=1)


@cli.command()
def proxy(
    *,
    version: Annotated[
        str,
        typer.Option("--version", "-V", help="Version string for the proxy packages."),
    ],
    output_dir: Annotated[
        pathlib.Path,
        typer.Option("--output-dir", "-o", help="Output directory for built distributions."),
    ] = pathlib.Path("dist"),
    sdist: Annotated[bool, typer.Option(help="Build source distributions.")] = True,
    wheel: Annotated[bool, typer.Option(help="Build wheel distributions.")] = True,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose output.")
    ] = False,
    packages: Annotated[
        list[str] | None,
        typer.Argument(
            help=(
                "Workspace member paths to build proxies for (e.g. 'model/common tools'). "
                "If omitted, proxies for all workspace members and the root package are built."
            ),
        ),
    ] = None,
) -> None:
    """
    Create empty proxy packages with the same names and a custom version.

    Useful for testing upload and download from private package indices without
    pulling in any real dependencies or code.
    """
    if not sdist and not wheel:
        typer.echo("Error: at least one of --sdist or --wheel must be enabled.", err=True)
        raise typer.Exit(code=1)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_targets = _collect_targets(packages)

    succeeded = 0
    failed: list[str] = []

    with tempfile.TemporaryDirectory(prefix="icon4py-proxy-") as tmp:
        tmp_root = pathlib.Path(tmp)

        for _source_dir, name in build_targets:
            pkg_dir = tmp_root / name
            pkg_dir.mkdir()

            pyproject_content = textwrap.dedent(f"""\
                [build-system]
                build-backend = "setuptools.build_meta"
                requires = ["setuptools>=61.0", "wheel>=0.40.0"]

                [project]
                name = "{name}"
                version = "{version}"
                description = "Empty proxy package for {name} (testing only)"
                requires-python = ">=3.10"

                [tool.setuptools]
                packages = []
            """)
            (pkg_dir / "pyproject.toml").write_text(pyproject_content)

            typer.echo(f"Building proxy {name} @ {version} ...")
            result = _run_uv_build(pkg_dir, output_dir, sdist=sdist, wheel=wheel, verbose=verbose)
            if result.returncode == 0:
                succeeded += 1
                typer.echo(f"  OK  {name}")
            else:
                failed.append(name)
                typer.echo(f"  FAIL  {name}", err=True)
                if not verbose and result.stderr:
                    sys.stderr.buffer.write(result.stderr)

    typer.echo(f"\n{succeeded}/{len(build_targets)} proxy packages built -> {output_dir}")
    if failed:
        typer.echo(f"Failed: {', '.join(failed)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    sys.exit(cli())

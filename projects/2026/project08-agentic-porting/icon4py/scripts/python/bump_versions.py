#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --only-group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Bump the version of all icon4py namespace packages to a new version."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
from typing import Annotated, Final

import typer
from helpers import common


def _find_versioned_package_dirs() -> list[pathlib.Path]:
    """Return directories of all pyproject.toml files in the repo that have a
    ``[tool.bumpversion]`` section, sorted with the repo root last."""
    dirs = []
    for pyproject in sorted(common.REPO_ROOT.rglob("pyproject.toml")):
        if ".venv" in pyproject.parts:
            continue
        if "[tool.bumpversion]" in pyproject.read_text():
            dirs.append(pyproject.parent)
    # Ensure repo root (if present) comes last so sub-packages bump first
    if common.REPO_ROOT in dirs:
        dirs.remove(common.REPO_ROOT)
        dirs.append(common.REPO_ROOT)
    return dirs


#: Pattern matching versioned icon4py cross-package dependency constraints, e.g.
#: ``icon4py-common>=0.0.6``, ``icon4py-common~=0.0.6``, or ``icon4py-tools~=0.0.6``.
_ICON4PY_DEP_CONSTRAINT_RE: Final = re.compile(
    r"(icon4py-[\w-]+(?:\[[\w,]+\])?)(~=|>=)([\d]+\.[\d]+\.[\d]+(?:(?:a|b|rc)\d+)?)"
)


cli = typer.Typer(no_args_is_help=True, help=__doc__)


def _detect_current_version(pkg_dirs: list[pathlib.Path]) -> str:
    """Read the current version from the first versioned package found."""
    for pkg_dir in pkg_dirs:
        pyproject = pkg_dir / "pyproject.toml"
        text = pyproject.read_text()
        m = re.search(
            r"^# managed by bump-my-version:\nversion = \"([\d.]+(?:(?:a|b|rc)\d+)?)\"",
            text,
            re.MULTILINE,
        )
        if m:
            return m.group(1)
    raise RuntimeError("Could not detect current version from any namespace package.")


def _bump_package(pkg_dir: pathlib.Path, new_version: str, dry_run: bool, verbose: bool) -> None:
    pyproject = pkg_dir / "pyproject.toml"
    if not pyproject.exists():
        typer.echo(f"  [skip] no pyproject.toml in {pkg_dir}", err=True)
        return

    cmd = [
        sys.executable,
        "-m",
        "bumpversion",
        "bump",
        "--new-version",
        new_version,
        "--allow-dirty",
        "--config-file",
        str(pyproject),
    ]
    if dry_run:
        cmd.append("--dry-run")
    if verbose:
        cmd.extend(["-v", "-v"])

    typer.echo(f"  Bumping {pkg_dir.relative_to(common.REPO_ROOT)} → {new_version}")
    result = subprocess.run(cmd, cwd=pkg_dir, capture_output=not verbose, check=False)
    if result.returncode != 0:
        err = result.stderr.decode() if result.stderr else ""
        typer.echo(f"  [ERROR] bump-my-version failed for {pkg_dir}:\n{err}", err=True)
        raise typer.Exit(code=1)


def _update_cross_package_constraints(
    current_version: str, new_version: str, dry_run: bool
) -> None:
    """Replace ``icon4py-*{op}{current_version}`` with ``icon4py-*{op}{new_version}``
    in every pyproject.toml across the repo (excluding .venv)."""
    for pyproject in sorted(common.REPO_ROOT.rglob("pyproject.toml")):
        # Skip anything inside the virtual environment
        if ".venv" in pyproject.parts:
            continue
        text = pyproject.read_text()
        new_text = _ICON4PY_DEP_CONSTRAINT_RE.sub(
            lambda m: (
                f"{m.group(1)}{m.group(2)}{new_version}"
                if m.group(3) == current_version
                else m.group(0)
            ),
            text,
        )
        if new_text != text:
            rel = pyproject.relative_to(common.REPO_ROOT)
            typer.echo(f"  Updating cross-package constraints in {rel}")
            if not dry_run:
                pyproject.write_text(new_text)


@cli.command()
def bump_versions(
    new_version: Annotated[str, typer.Argument(help="Target version, e.g. '0.1.0'")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would change without writing files")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Pass verbose flag to bump-my-version")
    ] = False,
) -> None:
    """Bump all namespace packages to NEW_VERSION and update cross-package constraints."""
    pkg_dirs = _find_versioned_package_dirs()
    current_version = _detect_current_version(pkg_dirs)
    typer.echo(f"Bumping all packages: {current_version} → {new_version}")
    if dry_run:
        typer.echo("  (dry-run mode — no files will be written)")

    typer.echo("\n[1/2] Running bump-my-version for each package:")
    for pkg_dir in pkg_dirs:
        _bump_package(pkg_dir, new_version, dry_run=dry_run, verbose=verbose)

    typer.echo("\n[2/2] Updating cross-package dependency constraints:")
    _update_cross_package_constraints(current_version, new_version, dry_run=dry_run)

    typer.echo("\nDone.")


if __name__ == "__main__":
    sys.exit(cli())

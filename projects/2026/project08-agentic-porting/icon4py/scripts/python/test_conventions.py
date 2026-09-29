#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --only-group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Manage test infrastructure."""

# TODO(egparedes): This module is still experimental. The code style is not great,
# it only has minimal documentation and it lacks proper tests, but it seems to
# work fine for the current needs, and it is already quite useful.

import ast
import contextlib
import enum
import functools
import os
import pathlib
import shlex
from typing import Annotated, NamedTuple

import pytest
import rich
import typer
from helpers import common


class ExitCode(enum.IntEnum):
    """Exit codes for the script."""

    MISSING_OR_INVALID_INIT_FILES = 1
    UNKNOWN_FIXTURE_REQUESTS = 2


cli = typer.Typer(
    name=__name__.split(".")[-1].replace("_", "-"),
    no_args_is_help=True,
    help="[DEPRECATED] " + __doc__,
)

# -- check-layout --
_INIT_PY_DEFAULT_CONTENT = (
    "\n".join(
        f"# {line}" if line.strip() else "#"
        for line in [*(common.REPO_ROOT / "HEADER.txt").read_text().splitlines(), "\n"]
    )
    + "\n"
)
_NS_INIT_PY_DEFAULT_CONTENT = (
    _INIT_PY_DEFAULT_CONTENT
    + "# Build on-the-fly a (legacy) namespace package for 'tests' using pkgutil\n"
    '__path__ = __import__("pkgutil").extend_path(__path__, __name__)\n'
)
_NS_INIT_PY_AST = ast.parse(_NS_INIT_PY_DEFAULT_CONTENT)


@cli.command(name="check-layout")
def check_layout(  # noqa: PLR0912 [too-many-branches]
    fix: Annotated[
        bool, typer.Option("--fix", "-f", help="Automatically create unknown __init__.py files.")
    ] = False,
):
    """Check if all 'tests' subpackages have proper '__init__.py' files."""
    root_dirs: list[pathlib.Path] = [common.REPO_ROOT / "model", common.REPO_ROOT / "tools"]
    violations = 0

    ast_dump = functools.partial(ast.dump, annotate_fields=False, include_attributes=False)

    for root_dir in root_dirs:
        prefix_len = len(root_dir.parts)

        for dir_path, _, file_names in root_dir.walk(top_down=True):
            if dir_path.name.startswith((".", "__")):
                # Skip hidden and special directories (e.g. '__pycache__')
                continue

            local_parts = dir_path.parts[prefix_len:]
            if "tests" in local_parts:
                if local_parts[-1] == "tests":  # 'tests' directory itself
                    rich.print(f"Checking '{'/'.join(local_parts)}' (namespace package)")

                    is_ns_init_ok = False
                    if "__init__.py" in file_names:
                        with contextlib.suppress(SyntaxError):
                            is_ns_init_ok = ast_dump(
                                ast.parse((dir_path / "__init__.py").read_text())
                            ) == ast_dump(_NS_INIT_PY_AST)

                    if not is_ns_init_ok:
                        violations += 1
                        rich.print(
                            "  [red]-> unknown or invalid '__init__.py' for namespace package[/red]"
                        )
                        if fix:
                            rich.print(f"  [yellow]-> Fixing '__init__.py' in {dir_path}[/yellow]")
                            (dir_path / "__init__.py").write_text(_NS_INIT_PY_DEFAULT_CONTENT)

                else:
                    rich.print(f"Checking '{'/'.join(local_parts)}'")

                    if "__init__.py" not in file_names:
                        violations += 1
                        rich.print("  [red]-> unknown '__init__.py' file[/red]")
                        if fix:
                            rich.print(
                                f"  [yellow]-> Creating '__init__.py' in {dir_path}[/yellow]"
                            )
                            (dir_path / "__init__.py").write_text(_INIT_PY_DEFAULT_CONTENT)

    rich.print()
    if violations:
        if fix:
            rich.print(f"[yellow]FIXED:[/yellow] {violations} issues have been fixed.")
        else:
            rich.print(f"[red]ERROR:[/red] {violations} issues have been found.")
            raise typer.Exit(code=ExitCode.MISSING_OR_INVALID_INIT_FILES)

    else:
        rich.print("[green]OK:[/green] Tests structure seems correct.")


# -- fixture-requests --
type FixtureRequestLocation = tuple[pathlib.Path, str]


class RequestedFixtures(NamedTuple):
    """A named tuple to hold all requested fixture names in a file."""

    all: set[str]
    unknown: set[str]


RequestedFixturesPerFile = dict[pathlib.Path, RequestedFixtures]


def _collect_fixture_requests(
    test_path: pathlib.Path | None = None, with_args: str = ""
) -> RequestedFixturesPerFile:
    """Collect all pytest fixtures from the specified test path."""
    collected_fixtures: RequestedFixturesPerFile = {}

    class CollectorPlugin:
        def pytest_collection_finish(self, session):
            for item in session.items:
                item_fixtures = set(item.fixturenames)

                if item.path not in collected_fixtures:
                    collected_fixtures[item.path] = RequestedFixtures(set(), set())
                all_, unknown = collected_fixtures[item.path]
                all_.update(item_fixtures)
                unknown.update(
                    item_fixtures - (item._fixtureinfo.name2fixturedefs.keys() | {"request"})
                )

    os.chdir(str(common.REPO_ROOT))
    test_path_arg = (
        [str(test_path.resolve().relative_to(common.REPO_ROOT))] if test_path is not None else []
    )
    pytest_cmd = [
        *test_path_arg,
        *shlex.split(with_args),
        "-q",
        "--no-header",
        "--no-summary",
        "--collect-only",
    ]
    pytest.main(
        pytest_cmd,
        plugins=[CollectorPlugin()],
    )
    return collected_fixtures


def _collect_fixtures_in_file(test_file_path: pathlib.Path) -> list[str]:
    """Parse a Python file and return the names of all pytest fixture defined inside."""

    try:
        tree = ast.parse(test_file_path.read_text(), filename=test_file_path)
    except SyntaxError:
        return []

    fixtures = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            fixtures.extend(
                node.name
                for decorator_node in node.decorator_list
                if (
                    isinstance(decorator_node, ast.Call)
                    and isinstance(decorator_node.func, ast.Attribute)
                    and decorator_node.func.attr == "fixture"
                    and isinstance(decorator_node.func.value, ast.Name)
                    and decorator_node.func.value.id == "pytest"
                )
                or (
                    isinstance(decorator_node, ast.Attribute)
                    and decorator_node.attr == "fixture"
                    and isinstance(decorator_node.value, ast.Name)
                    and decorator_node.value.id == "pytest"
                )
            )
    return fixtures


def _collect_fixture_files(root_dir: pathlib.Path) -> list[pathlib.Path]:
    """Find all Python files in 'fixtures' folders or named 'fixtures.py'."""
    fixture_pkgs = set()
    fixture_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "/site-packages/" in dirpath:
            continue  # Skip virtual environments
        if (dirpath.endswith("fixtures") or dirpath in fixture_pkgs) and "__init__.py" in filenames:
            fixture_pkgs.update(f"{dirpath}/{d}" for d in dirnames)
            fixture_files.extend(
                pathlib.Path(dirpath) / fname for fname in filenames if fname.endswith(".py")
            )
            if dirpath in fixture_pkgs:
                fixture_pkgs.remove(dirpath)
        elif "fixtures.py" in filenames:
            fixture_files.append(pathlib.Path(dirpath) / "fixtures.py")
    return fixture_files


def _collect_fixtures(root_dir: pathlib.Path) -> dict[str, list[pathlib.Path]]:
    """Collect all pytest fixtures in the project."""
    fixtures_dict = {}
    fixture_files = _collect_fixture_files(root_dir)
    for file_path in fixture_files:
        fixtures = _collect_fixtures_in_file(file_path)
        for fixture in fixtures:
            fixtures_dict.setdefault(fixture, []).append(file_path.relative_to(root_dir))
    return fixtures_dict


def _find_closest_fixture_import_path(
    test_file_path: pathlib.Path, fixture_definitions: list[pathlib.Path]
) -> str | None:
    """Find the closest import path for a fixture definition relative to the test file."""
    test_file_path = test_file_path.resolve().relative_to(common.REPO_ROOT)
    test_file_component = test_file_path.parts[test_file_path.parts.index("tests") + 1]
    up_levels = 100
    fixture_import = None

    for def_path in fixture_definitions:
        if (
            "tests" in def_path.parts
            and def_path.parts[def_path.parts.index("tests") + 1] == test_file_component
        ):
            relative_path = def_path.relative_to(test_file_path.parent, walk_up=True)
            if (levels := relative_path.parts.count("..")) < up_levels:
                up_levels = levels
                fixture_import = "".join(
                    "." if part == ".." else part.split(".")[0] for part in relative_path.parts
                )

        elif fixture_import is None:
            fixture_import = ".".join(
                p.split(".")[0] for p in def_path.parts[def_path.parts.index("src") + 1 :]
            )

    return fixture_import


def _fix_fixture_requests(
    fixture_requests: RequestedFixturesPerFile,
    test_path: pathlib.Path,
) -> tuple[list[FixtureRequestLocation], list[FixtureRequestLocation]]:
    """Fix unknown fixture requests by adding explicit imports."""
    rich.print("[yellow]Adding missing fixture imports...[/yellow]")

    fixed = []
    errors = []
    fixture_definitions = _collect_fixtures(common.REPO_ROOT)

    for path, (_all_fixtures, unknown_fixtures) in fixture_requests.items():
        if not unknown_fixtures:
            continue

        rich.print(f"- '{path}':")
        new_imports = []
        for fixture in unknown_fixtures:
            rich.print(f"    + fixture: '{fixture}'")
            matching_defs = {", ".join(f"'{f!s}'" for f in fixture_definitions.get(fixture, []))}
            rich.print(f"    + matching definitions: {matching_defs} ")
            if fixture not in fixture_definitions or not (
                import_path := _find_closest_fixture_import_path(path, fixture_definitions[fixture])
            ):
                errors += (path, fixture)
                continue
            assert import_path
            fixed += (path, fixture)
            new_imports.append(f"from {import_path} import {fixture}\n")

        lines = path.read_text().splitlines(keepends=True)
        i = 0
        for i in range(len(lines)):
            line = lines[i].rstrip()
            if line and not (
                line.startswith("import ")
                or line.startswith("from ")
                or line.startswith("#")
                or line.startswith("(")
                or line.startswith(")")
                or line.startswith("    ")
            ):
                break

        path.write_text("".join([*lines[:i], *new_imports, "\n", *lines[i:]]))

    return fixed, errors


@cli.command(name="fixture-requests", help=_collect_fixture_requests.__doc__)
def fixture_requests(
    test_path: Annotated[
        pathlib.Path | None,
        typer.Argument(help="The path to collect fixture requests for (default: pytest default)."),
    ] = None,
    with_args: Annotated[
        str, typer.Option("--with-args", "-w", help="Pass additional arguments to pytest.")
    ] = "",
    fix: Annotated[
        bool,
        typer.Option(
            "--fix",
            "-f",
            help="Automatically create explicit imports for missing fixtures in test files.",
        ),
    ] = False,
):
    fixture_requests: RequestedFixturesPerFile = _collect_fixture_requests(test_path, with_args)

    report = []
    errors: list[FixtureRequestLocation] = []
    fixes = None
    for path, (all_, unknown) in fixture_requests.items():
        all_list = sorted(all_)
        unknown_list = sorted(unknown)
        report.append(
            f"- {path}:\n"
            f"    + {len(all_)} requested{':' if all_ else ''} {', '.join(all_list)}\n"
            f"    + {len(unknown)} unknown{':' if unknown else ''} {', '.join(unknown_list)}\n"
        )
        if unknown_list:
            errors.extend((path, fixture) for fixture in unknown_list)

    rich.print("\n".join(report))

    if errors and fix:
        fixes, errors = _fix_fixture_requests(fixture_requests, test_path)

    if errors:
        rich.print(
            f"[red]ERROR:[/red] {len(errors)} fixtures were requested but not found in the test files."
        )
        rich.print(errors)
        raise typer.Exit(code=ExitCode.MISSING_OR_INVALID_INIT_FILES)
    elif fixes:
        rich.print(
            f"[yellow]Fixed {len(fixes)} fixture requests in {len(fixture_requests)} test files.[/yellow]"
        )
    else:
        rich.print("[green]OK:[/green] No unknown fixture requests found in the test files.")


if __name__ == "__main__":
    cli()

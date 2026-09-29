#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --only-group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate GitLab CI child pipeline YAML from pipeline variables.

Reads the pipeline variables (SESSIONS, MODEL_SUBPACKAGES, MODEL_MPI_SUBPACKAGES,
BACKENDS, LEVELS, GRIDS, MODEL_SUBSETS, TOOLS_SUBSETS) or corresponding command-line options
and writes a child pipeline that includes ``ci/base.yml`` and instantiates only
the test jobs whose matrix entries match the requested filter and collect at
least one test.

Each SESSION value maps to a single job template that corresponds to a nox
session. Parameters like MODEL_SUBPACKAGES and MODEL_SUBSETS are passed as nox
session parameters.

This script exists because using rules:if with regexes that are dynamically
generated does not work (gitlab does not expand variables in patterns). This
script does the filtering at generation time instead.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Annotated

import typer
import yaml


cli = typer.Typer(no_args_is_help=True, help=__doc__)

ALL_SESSIONS = ["model", "model_mpi", "tools"]
ALL_MODEL_SUBSETS = ["stencils", "datatest", "basic"]
ALL_MODEL_MPI_SUBSETS = ["basic", "datatest"]
ALL_MODEL_SUBPACKAGES = [
    "advection",
    "diffusion",
    "dycore",
    "microphysics",
    "muphys",
    "common",
    "driver",
    "standalone_driver",
]

ALL_MODEL_MPI_SUBPACKAGES = [
    "advection",
    "diffusion",
    "dycore",
    "common",
    "standalone_driver",
]
ALL_BACKENDS = ["embedded", "dace_cpu", "dace_gpu", "gtfn_cpu", "gtfn_gpu"]
ALL_GRIDS = ["simple", "icon_regional", "icon_global"]
# Note that ALL_LEVELS does _not_ include "any", even though it's a valid option
# for --level, because the implicit "all" generates test jobs for "unit" and
# "integration". We don't want "any", "unit", and "integration" all to be
# enabled since that would run the same tests in multiple jobs.
# TODO(msimberg): Revisit this to see if the levels, names, or something else
# should be changed to simplify this.
ALL_LEVELS = ["unit", "integration", "validation"]
ALL_TOOLS_SUBSETS = ["datatest", "unittest"]

# Collection tuning. The per-cell timeout should be generous enough for the
# first cold import of icon4py/GT4Py; the overall collection run is bounded
# by the Slurm job time limit (ci/base.yml). Values are intentionally
# conservative and can be reduced once real timings are available.
_COLLECTION_TIMEOUT_SECONDS = 5 * 60
_COLLECTION_MAX_WORKERS = int(os.environ.get("ICON4PY_NOX_COLLECTION_WORKERS", "8"))


def _parse_list(raw: str | None) -> list[str]:
    """Parse a colon- or comma-separated string into a list of tokens.

    Colons are supported as separators because commas can't be used as
    separators in CSCS CI variables when triggering jobs. The comma is reserved
    for separating pipeline names in cscs-ci run pipeline1,pipeline2.
    """
    if not raw:
        return []
    return [x.strip() for x in re.split(r"[,:]", raw) if x.strip()]


def _intersect(constraint: list[str], candidates: list[str]) -> list[str]:
    """Return *candidates* that are in *constraint*, preserving *candidates* order."""
    constraint_set = set(constraint)
    return [v for v in candidates if v in constraint_set]


def _validate_tokens(name: str, tokens: list[str], valid: list[str]) -> None:
    """Validate that all tokens are members of valid.

    Exits with a descriptive error message and status 1 if any token is not
    recognised.
    """
    invalid = [t for t in tokens if t not in valid]
    if invalid:
        accepted = ", ".join(sorted(valid))
        print(
            f"ERROR: invalid {name} values: {', '.join(invalid)}",
            file=sys.stderr,
        )
        print(f"Accepted values: {accepted}", file=sys.stderr)
        sys.exit(1)


def _resolve_filter(cli_value: str | None, env_var: str, *, default: list[str]) -> list[str]:
    """Resolve a filter value from CLI arg, env var, or built-in default.

    When *cli_value* is provided (including empty string) it takes
    precedence.  Otherwise the environment variable is checked,
    falling back to *default*.

    The token ``all`` expands to the full *default* list.  It must not be
    combined with other values.
    """
    if cli_value is not None:
        tokens = _parse_list(cli_value)
    else:
        env_parsed = _parse_list(os.environ.get(env_var))
        if env_parsed:
            tokens = env_parsed
        else:
            return list(default)

    if "all" in tokens:
        if len(tokens) > 1:
            print(
                f"ERROR: '{env_var}' contains 'all' but also other values. "
                "Use 'all' alone or list individual values.",
                file=sys.stderr,
            )
            sys.exit(1)
        return list(default)

    return tokens


def _nox_session_name(base: str, params: str) -> str:
    """Return the nox session name used for collection.

    *params* must be the comma-separated parameter values exactly as nox
    renders them in session names (e.g. ``"basic, common"``), not keyword
    assignments.

    When collection runs with ``ICON4PY_NOX_USE_ACTIVE_VENV=1`` nox ignores the
    ``python=`` parametrization and names sessions without a Python-version
    suffix.
    """
    return f"{base}({params})"


def _collection_env() -> dict[str, str]:
    """Return environment variables for the offline collection runs."""
    env: dict[str, str] = {
        "ICON4PY_NOX_USE_ACTIVE_VENV": "1",
    }
    # Point uv to the project venv, which contains pytest and icon4py.
    # In the CI image the venv is at /icon4py/.venv and is exposed via
    # ICON4PY_COLLECTION_VENV; locally it is discovered from pytest in PATH.
    if configured_venv := os.environ.get("ICON4PY_COLLECTION_VENV"):
        env["VIRTUAL_ENV"] = configured_venv
    elif pytest_path := shutil.which("pytest"):
        env["VIRTUAL_ENV"] = str(pathlib.Path(pytest_path).parent.parent)
    else:
        raise RuntimeError(
            "Cannot determine collection virtualenv. Set ICON4PY_COLLECTION_VENV or ensure pytest is on PATH."
        )
    return env


def _log_collection_output(
    prefix: str, cmd: list[str], returncode: int, stdout: str, stderr: str
) -> None:
    """Print nox/pytest collection output to stderr for diagnostics."""
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"{prefix}: {' '.join(cmd)}", file=sys.stderr)
    print(f"exit code: {returncode}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    output = (stdout + "\n" + stderr).strip()
    if output:
        print(output, file=sys.stderr)
    else:
        print("<no output>", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)


def _run_nox_collection(
    session_name: str,
    pytest_args: list[str],
    env: dict[str, str],
    timeout: float,
) -> bool:
    """Run a nox session with --collect-only and return whether to keep the cell.

    Returns True when nox exits 0 (the cell collected at least one runnable
    test). Returns False when nox exits 1 (the cell collected zero tests).

    Raises on any other nox exit code, subprocess timeout, or OSError.
    """
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "--active",
        "--with",
        "nox",
        "nox",
        "-s",
        session_name,
        "--",
        *pytest_args,
    ]
    full_env = os.environ.copy()
    full_env.update(env)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=full_env,
        check=False,
    )

    if result.returncode == 0:
        return True

    combined = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode == 1 and re.search(
        r"(?:exit code|returned non-zero exit code) 5\b", combined
    ):
        _log_collection_output(
            "DEBUG: nox collection returned exit 1 (zero tests)",
            cmd,
            result.returncode,
            result.stdout,
            result.stderr,
        )
        return False

    _log_collection_output(
        "ERROR: nox collection failed",
        cmd,
        result.returncode,
        result.stdout,
        result.stderr,
    )
    raise subprocess.CalledProcessError(
        result.returncode, cmd, output=combined, stderr=result.stderr
    )


@dataclass(frozen=True)
class _MatrixCell:
    """A single matrix expansion cell together with its collection recipe."""

    job_name: str
    extends: str
    variables: dict[str, str]
    matrix: dict[str, str]
    session: str
    pytest_args: list[str]


def _model_cells(
    subpackages: list[str],
    backends: list[str],
    grids: list[str],
    levels: list[str],
    subsets: list[str],
) -> list[_MatrixCell]:
    """Build collection cells for the serial model test sessions."""
    cells: list[_MatrixCell] = []

    if "stencils" in subsets and grids:
        for subpackage in subpackages:
            for backend in backends:
                for grid in grids:
                    cells.append(  # noqa: PERF401
                        _MatrixCell(
                            job_name="test_model_stencils_aarch64",
                            extends=".test_model_aarch64",
                            variables={"MODEL_SUBSET": "stencils"},
                            matrix={
                                "MODEL_SUBPACKAGE": subpackage,
                                "BACKEND": backend,
                                "GRID": grid,
                            },
                            session=_nox_session_name("test_model", f"stencils, {subpackage}"),
                            pytest_args=[
                                "--collect-only",
                                "-n0",
                                "-p",
                                "no:tach",
                                f"--backend={backend}",
                                f"--grid={grid}",
                            ],
                        )
                    )

    for subset in ("datatest", "basic"):
        if subset not in subsets or not levels:
            continue
        for subpackage in subpackages:
            for backend in backends:
                for level in levels:
                    pytest_args = [
                        "--collect-only",
                        "-n0",
                        "-p",
                        "no:tach",
                        f"--backend={backend}",
                        f"--level={level}",
                    ]
                    cells.append(
                        _MatrixCell(
                            job_name=f"test_model_{subset}_aarch64",
                            extends=".test_model_aarch64",
                            variables={"MODEL_SUBSET": subset},
                            matrix={
                                "MODEL_SUBPACKAGE": subpackage,
                                "BACKEND": backend,
                                "LEVEL": level,
                            },
                            session=_nox_session_name("test_model", f"{subset}, {subpackage}"),
                            pytest_args=pytest_args,
                        )
                    )

    return cells


def _tools_cells(selections: list[str]) -> list[_MatrixCell]:
    """Build collection cells for the tools/bindings test session."""
    cells: list[_MatrixCell] = []
    for selection in selections:
        pytest_args = ["--collect-only", "-n0", "-p", "no:tach"]
        cells.append(
            _MatrixCell(
                job_name="test_tools_aarch64",
                extends=".test_tools_aarch64",
                variables={},
                matrix={"SELECTION": selection},
                session=_nox_session_name("test_tools_and_bindings", selection),
                pytest_args=pytest_args,
            )
        )
    return cells


def _model_mpi_cells(
    subpackages: list[str],
    backends: list[str],
    levels: list[str],
    subsets: list[str],
) -> list[_MatrixCell]:
    """Build collection cells for the MPI model test sessions."""
    cells: list[_MatrixCell] = []
    if not subpackages or not backends or not levels:
        return cells
    for subset in subsets:
        for subpackage in subpackages:
            for backend in backends:
                for level in levels:
                    pytest_args = [
                        "--collect-only",
                        "-n0",
                        "-p",
                        "no:tach",
                        f"--backend={backend}",
                        f"--level={level}",
                    ]
                    cells.append(
                        _MatrixCell(
                            job_name=f"test_model_mpi_{subset}_aarch64",
                            extends=".test_model_mpi_aarch64",
                            variables={"SELECTION": subset},
                            matrix={
                                "MODEL_MPI_SUBPACKAGE": subpackage,
                                "BACKEND": backend,
                                "LEVEL": level,
                            },
                            session=_nox_session_name("test_model_mpi", f"{subset}, {subpackage}"),
                            pytest_args=pytest_args,
                        )
                    )
    return cells


def _collect_cells(cells: list[_MatrixCell]) -> tuple[list[_MatrixCell], list[_MatrixCell]]:
    """Run collection for every cell in parallel and return kept/dropped cells.

    Cells where nox exits 0 are kept. Cells where nox exits 1 are dropped
    because the corresponding matrix cell contains no tests.
    Any collection failure, subprocess timeout, or non-0/1 exit code aborts
    pipeline generation.
    """
    if os.environ.get("ICON4PY_CI_SKIP_COLLECTION"):
        return cells, []

    if not cells:
        return cells, []

    env = _collection_env()
    results: list[bool] = [False] * len(cells)

    with ThreadPoolExecutor(max_workers=_COLLECTION_MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                _run_nox_collection,
                cell.session,
                cell.pytest_args,
                env,
                _COLLECTION_TIMEOUT_SECONDS,
            ): i
            for i, cell in enumerate(cells)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    kept = [cell for cell, ok in zip(cells, results) if ok]
    dropped = [cell for cell, ok in zip(cells, results) if not ok]
    return kept, dropped


def _build_pipeline(cells: list[_MatrixCell]) -> dict:
    """Build the child pipeline dict from the surviving matrix cells."""
    pipeline: dict = {"include": [{"local": "ci/base.yml"}]}
    jobs: dict[str, dict] = {}
    for cell in cells:
        job = jobs.setdefault(
            cell.job_name,
            {
                "extends": cell.extends,
                "variables": dict(cell.variables),
                "parallel": {"matrix": []},
            },
        )
        job["parallel"]["matrix"].append(dict(cell.matrix))

    pipeline.update(jobs)

    return pipeline


def _format_cell(cell: _MatrixCell) -> str:
    matrix = ", ".join(f"{key}={value}" for key, value in cell.matrix.items())
    return f"  {cell.job_name}: {matrix}"


def _print_collection_summary(
    *,
    sessions: list[str],
    model_subpackages: list[str],
    model_mpi_subpackages: list[str],
    model_subsets: list[str],
    model_mpi_subsets: list[str],
    tools_subsets: list[str],
    backends: list[str],
    levels: list[str],
    grids: list[str],
    kept: list[_MatrixCell],
    dropped: list[_MatrixCell],
) -> None:
    """Print a human-readable summary of the collection results to stderr."""
    print("INFO: CI pipeline collection summary", file=sys.stderr)
    print(f"  sessions: {sessions}", file=sys.stderr)
    if model_subpackages:
        print(f"  model_subpackages: {model_subpackages}", file=sys.stderr)
    if model_mpi_subpackages:
        print(f"  model_mpi_subpackages: {model_mpi_subpackages}", file=sys.stderr)
    if model_subsets:
        print(f"  model_subsets: {model_subsets}", file=sys.stderr)
    if model_mpi_subsets:
        print(f"  model_mpi_subsets: {model_mpi_subsets}", file=sys.stderr)
    if tools_subsets:
        print(f"  tools_subsets: {tools_subsets}", file=sys.stderr)
    if backends:
        print(f"  backends: {backends}", file=sys.stderr)
    if levels:
        print(f"  levels: {levels}", file=sys.stderr)
    if grids:
        print(f"  grids: {grids}", file=sys.stderr)
    print(f"  eligible cells: {len(kept) + len(dropped)}", file=sys.stderr)
    print(f"  selected cells: {len(kept)}", file=sys.stderr)
    for cell in kept:
        print(_format_cell(cell), file=sys.stderr)
    print(
        f"  unselected cells: {len(dropped)} "
        "(dropped because collection found zero tests in the matrix cell)",
        file=sys.stderr,
    )
    for cell in dropped:
        print(_format_cell(cell), file=sys.stderr)


def _generate_child_pipeline(
    *,
    sessions: str | None = None,
    model_subsets: str | None = None,
    model_subpackages: str | None = None,
    model_mpi_subpackages: str | None = None,
    model_mpi_subsets: str | None = None,
    tools_subsets: str | None = None,
    backends: str | None = None,
    levels: str | None = None,
    grids: str | None = None,
) -> str:
    """Return the child pipeline YAML as a string.

    Each subset is expanded into a separate job definition with a
    ``parallel:matrix`` for the remaining dimensions. Matrix entries that
    collect zero tests are omitted.

    GitLab limits each ``parallel:matrix`` to 200 instances; callers must
    ensure the expanded matrix does not exceed this limit.
    """
    requested_sessions = _resolve_filter(sessions, "SESSIONS", default=ALL_SESSIONS)
    _validate_tokens("SESSIONS", requested_sessions, ALL_SESSIONS)

    requested_model_subsets = _resolve_filter(
        model_subsets, "MODEL_SUBSETS", default=ALL_MODEL_SUBSETS
    )
    _validate_tokens("MODEL_SUBSETS", requested_model_subsets, ALL_MODEL_SUBSETS)

    requested_model_subpackages = _resolve_filter(
        model_subpackages,
        "MODEL_SUBPACKAGES",
        default=ALL_MODEL_SUBPACKAGES,
    )
    _validate_tokens("MODEL_SUBPACKAGES", requested_model_subpackages, ALL_MODEL_SUBPACKAGES)

    requested_model_mpi_subpackages = _resolve_filter(
        model_mpi_subpackages,
        "MODEL_MPI_SUBPACKAGES",
        default=ALL_MODEL_MPI_SUBPACKAGES,
    )
    _validate_tokens(
        "MODEL_MPI_SUBPACKAGES", requested_model_mpi_subpackages, ALL_MODEL_MPI_SUBPACKAGES
    )

    requested_model_mpi_subsets = _resolve_filter(
        model_mpi_subsets, "MODEL_MPI_SUBSETS", default=ALL_MODEL_MPI_SUBSETS
    )
    _validate_tokens("MODEL_MPI_SUBSETS", requested_model_mpi_subsets, ALL_MODEL_MPI_SUBSETS)

    requested_backends = _resolve_filter(backends, "BACKENDS", default=ALL_BACKENDS)
    _validate_tokens("BACKENDS", requested_backends, ALL_BACKENDS)

    requested_levels = _resolve_filter(levels, "LEVELS", default=ALL_LEVELS)
    _validate_tokens("LEVELS", requested_levels, ALL_LEVELS)

    requested_grids = _resolve_filter(grids, "GRIDS", default=ALL_GRIDS)
    _validate_tokens("GRIDS", requested_grids, ALL_GRIDS)

    requested_tools_subsets = _resolve_filter(
        tools_subsets, "TOOLS_SUBSETS", default=ALL_TOOLS_SUBSETS
    )
    _validate_tokens("TOOLS_SUBSETS", requested_tools_subsets, ALL_TOOLS_SUBSETS)

    cells: list[_MatrixCell] = []

    if "model" in requested_sessions:
        cells.extend(
            _model_cells(
                subpackages=_intersect(requested_model_subpackages, ALL_MODEL_SUBPACKAGES),
                backends=_intersect(requested_backends, ALL_BACKENDS),
                grids=_intersect(requested_grids, ALL_GRIDS),
                levels=_intersect(requested_levels, ALL_LEVELS),
                subsets=_intersect(requested_model_subsets, ALL_MODEL_SUBSETS),
            )
        )

    if "tools" in requested_sessions:
        cells.extend(
            _tools_cells(
                selections=_intersect(requested_tools_subsets, ALL_TOOLS_SUBSETS),
            )
        )

    if "model_mpi" in requested_sessions:
        cells.extend(
            _model_mpi_cells(
                subpackages=_intersect(requested_model_mpi_subpackages, ALL_MODEL_MPI_SUBPACKAGES),
                backends=_intersect(requested_backends, ALL_BACKENDS),
                levels=_intersect(requested_levels, ALL_LEVELS),
                subsets=_intersect(requested_model_mpi_subsets, ALL_MODEL_MPI_SUBSETS),
            )
        )

    kept_cells, dropped_cells = _collect_cells(cells)

    _print_collection_summary(
        sessions=requested_sessions,
        model_subpackages=requested_model_subpackages,
        model_mpi_subpackages=requested_model_mpi_subpackages,
        model_subsets=requested_model_subsets,
        model_mpi_subsets=requested_model_mpi_subsets,
        tools_subsets=requested_tools_subsets,
        backends=requested_backends,
        levels=requested_levels,
        grids=requested_grids,
        kept=kept_cells,
        dropped=dropped_cells,
    )

    pipeline = _build_pipeline(kept_cells)

    test_jobs = [k for k in pipeline if k != "include"]
    if not test_jobs:
        print(
            "ERROR: no test jobs matched the filter; all eligible matrix cells "
            "were dropped because collection found zero tests in each of them.",
            file=sys.stderr,
        )
        sys.exit(1)

    return yaml.safe_dump(pipeline)


@cli.command()
def generate_ci_pipeline(  # noqa: PLR0917 [too-many-positional-arguments]
    sessions: Annotated[
        str | None,
        typer.Option(
            "--sessions",
            help="Colon/comma-separated nox session filter (model, model_mpi, tools)",
        ),
    ] = None,
    model_subpackages: Annotated[
        str | None,
        typer.Option(
            "--model-subpackages",
            help="Colon/comma-separated model subpackage filter",
        ),
    ] = None,
    model_mpi_subpackages: Annotated[
        str | None,
        typer.Option(
            "--model-mpi-subpackages",
            help="Colon/comma-separated MPI subpackage filter",
        ),
    ] = None,
    model_mpi_subsets: Annotated[
        str | None,
        typer.Option(
            "--model-mpi-subsets",
            help="Colon/comma-separated MPI test subset filter (basic, datatest)",
        ),
    ] = None,
    model_subsets: Annotated[
        str | None,
        typer.Option(
            "--model-subsets",
            help="Colon/comma-separated model test subset filter (stencils, datatest, basic)",
        ),
    ] = None,
    tools_subsets: Annotated[
        str | None,
        typer.Option(
            "--tools-subsets",
            help="Colon/comma-separated tools/bindings test subset filter (datatest, unittest)",
        ),
    ] = None,
    backends: Annotated[
        str | None,
        typer.Option("--backends", help="Colon/comma-separated backend filter"),
    ] = None,
    levels: Annotated[
        str | None,
        typer.Option("--levels", help="Colon/comma-separated level filter"),
    ] = None,
    grids: Annotated[
        str | None,
        typer.Option("--grids", help="Colon/comma-separated grid filter"),
    ] = None,
) -> None:
    """Generate child pipeline YAML to stdout.

    Reads pipeline variables from environment or CLI options and writes
    a child pipeline to standard output that includes only the test jobs
    matching the requested filter.
    """
    sys.stdout.write(
        _generate_child_pipeline(
            sessions=sessions,
            model_subpackages=model_subpackages,
            model_mpi_subpackages=model_mpi_subpackages,
            model_mpi_subsets=model_mpi_subsets,
            model_subsets=model_subsets,
            tools_subsets=tools_subsets,
            backends=backends,
            levels=levels,
            grids=grids,
        )
    )


if __name__ == "__main__":
    sys.exit(cli())

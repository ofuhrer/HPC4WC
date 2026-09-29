# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Final, Literal, TypedDict, get_args

import nox


# -- nox configuration --
def _use_active_venv() -> bool:
    """Return True when nox should run in the active Python environment."""
    return os.environ.get("ICON4PY_NOX_USE_ACTIVE_VENV") == "1"


nox.options.sessions = ["test_model", "test_tools_and_bindings"]
nox.options.default_venv_backend = "uv"


class _VenvBackendKwargs(TypedDict, total=False):
    venv_backend: str


# When running inside the already-active project venv, avoid creating per-session
# venvs. Applied only to test sessions; benchmark/bencher sessions keep their
# default uv backend.
_VENV_BACKEND_KWARG: Final[_VenvBackendKwargs] = (
    {"venv_backend": "none"} if _use_active_venv() else {}
)
NO_TESTS_COLLECTED_EXIT_CODE: Final = 5

_rank = (
    os.environ.get("PMI_RANK")
    or os.environ.get("OMPI_COMM_WORLD_RANK")
    or os.environ.get("SLURM_PROCID")
)
if _rank is not None:
    nox.options.envdir = f".nox/mpi-rank-{_rank}"


# -- Parameter sets --
type ModelSubpackagePath = Literal[
    "atmosphere/advection",
    "atmosphere/diffusion",
    "atmosphere/dycore",
    "atmosphere/subgrid_scale_physics/microphysics",
    "atmosphere/subgrid_scale_physics/muphys",
    "common",
    "driver",
    "standalone_driver",
    "testing",
]
MODEL_SUBPACKAGE_PATHS: Final[Sequence[nox.Param]] = [
    nox.param(arg, id=arg.split("/")[-1]) for arg in get_args(ModelSubpackagePath.__value__)
]

type ModelTestsSubset = Literal["datatest", "stencils", "basic"]
MODEL_TESTS_SUBSETS: Final[Sequence[nox.Param]] = [
    nox.param(arg, id=arg, tags=[arg]) for arg in get_args(ModelTestsSubset.__value__)
]
# Stencil tests are by definition serial
MODEL_MPI_TESTS_SUBSETS: Final[Sequence[nox.Param]] = [
    nox.param(arg, id=arg, tags=[arg])
    for arg in get_args(ModelTestsSubset.__value__)
    if arg != "stencils"
]
type ToolsBindingsTestsSubset = Literal["datatest", "unittest"]
TOOLS_BINDINGS_TESTS_SUBSETS: Final[Sequence[nox.Param]] = [
    nox.param(arg, id=arg, tags=[arg]) for arg in get_args(ToolsBindingsTestsSubset.__value__)
]
SUPPORTED_PYTHON_VERSIONS: Final[Sequence[str]] = ["3.12", "3.13", "3.14"]


# -- nox sessions --
# Model benchmark sessions
# TODO(egparedes): Add backend parameter
# TODO(edopao,egparedes): Change 'extras' back to 'all' once mpi4py can be compiled with hpc_sdk
@nox.session(python=SUPPORTED_PYTHON_VERSIONS)
def benchmark_model(session: nox.Session) -> None:
    """Run pytest benchmarks."""
    _install_session_venv(session, extras=["io", "testing"], groups=["test"])

    session.run(
        *f"pytest \
        -v \
        -m continuous_benchmarking \
        --benchmark-warmup=on \
        --benchmark-warmup-iterations=10 \
        --benchmark-json=pytest_benchmark_results_{session.python}.json \
        ./model".split(),
        *session.posargs,
    )


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, requires=["benchmark_model-{python}"])
def __bencher_baseline_CI(session: nox.Session) -> None:
    """
    Run pytest benchmarks and upload them using Bencher (https://bencher.dev/) (cloud or self-hosted).
    This session is used only on the main branch to create the historical baseline.
    The historical baseline is used to compare the performance of the code in the PRs.
    Alerts are raised if there is performance regression according to the thresholds.
    Note: This session is intended to be run from the CI only -bencher and suitable env vars are needed-.
    """
    session.run(
        *f"bencher run \
        --threshold-measure latency \
        --threshold-test percentage \
        --threshold-max-sample-size 64 \
        --threshold-upper-boundary 0.1 \
        --thresholds-reset \
        --err \
        --file pytest_benchmark_results_{session.python}.json".split(),
        env={
            "BENCHER_PROJECT": os.environ[
                "BENCHER_PROJECT"
            ].strip(),  # defined in https://cicd-ext-mw.cscs.ch
            "BENCHER_BRANCH": "main",
            "BENCHER_TESTBED": f"{os.environ['RUNNER']}:{os.environ['SYSTEM_TAG']}:{os.environ['BACKEND']}:{os.environ['GRID']}",
            "BENCHER_ADAPTER": "python_pytest",
            "BENCHER_HOST": os.environ[
                "BENCHER_HOST"
            ].strip(),  # defined in https://cicd-ext-mw.cscs.ch
            "BENCHER_API_TOKEN": os.environ["BENCHER_API_TOKEN"].strip(),
        },
        external=True,
        silent=True,
    )


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, requires=["benchmark_model-{python}"])
def __bencher_feature_branch_CI(session: nox.Session) -> None:
    """
    Run pytest benchmarks and upload them using Bencher (https://bencher.dev/) (cloud or self-hosted).
    This session compares the performance of the feature branch with the historical baseline (as built from __bencher_baseline_CI session).
    Alerts are raised if the performance of the feature branch is worse than the historical baseline (according to the thresholds).
    Note: This session is intended to be run from the CI only -bencher and suitable env vars are needed-.
    """
    bencher_testbed = f"{os.environ['RUNNER']}:{os.environ['SYSTEM_TAG']}:{os.environ['BACKEND']}:{os.environ['GRID']}"
    session.run(
        *f"bencher run \
        --start-point main \
        --start-point-clone-thresholds \
        --start-point-reset \
        --err \
        --github-actions {os.environ['GD_COMMENT_TOKEN']} \
        --ci-number {os.environ['PR_ID']} \
        --ci-id run-{bencher_testbed.replace(':', '_')}-{int(datetime.now().strftime('%Y%m%d%H%M%S%f'))} \
        --file pytest_benchmark_results_{session.python}.json".split(),
        env={
            "BENCHER_PROJECT": os.environ[
                "BENCHER_PROJECT"
            ].strip(),  # defined in https://cicd-ext-mw.cscs.ch
            "BENCHER_BRANCH": os.environ["FEATURE_BRANCH"].strip(),
            "BENCHER_TESTBED": bencher_testbed,
            "BENCHER_ADAPTER": "python_pytest",
            "BENCHER_HOST": os.environ[
                "BENCHER_HOST"
            ].strip(),  # defined in https://cicd-ext-mw.cscs.ch
            "BENCHER_API_TOKEN": os.environ["BENCHER_API_TOKEN"].strip(),
        },
        external=True,
        silent=True,
    )


# Model test sessions
# TODO(egparedes): Add backend parameter
# TODO(edopao,egparedes): Change 'extras' back to 'all' once mpi4py can be compiled with hpc_sdk
@nox.session(python=SUPPORTED_PYTHON_VERSIONS, **_VENV_BACKEND_KWARG)
@nox.parametrize("subpackage", MODEL_SUBPACKAGE_PATHS)
@nox.parametrize("selection", MODEL_TESTS_SUBSETS)
def test_model(
    session: nox.Session, selection: ModelTestsSubset, subpackage: ModelSubpackagePath
) -> None:
    """Run tests for selected icon4py model subpackages."""
    _install_session_venv(session, extras=["fortran", "io", "testing"], groups=["test"])

    pytest_args = _selection_to_pytest_args(selection)
    success_codes = (
        [0] if "--collect-only" in session.posargs else [0, NO_TESTS_COLLECTED_EXIT_CODE]
    )
    with session.chdir(f"model/{subpackage}"):
        session.run(
            *f"pytest -sv --benchmark-disable -n {os.environ.get('NUM_PROCESSES', 'auto')}".split(),
            *pytest_args,
            "tests",
            *session.posargs,
            success_codes=success_codes,
        )


# MPI test session. Per-rank venv isolation is handled automatically by setting
# nox.options.envdir to ".nox/mpi-rank-<rank>" at import time when an MPI
# rank variable is present. Note that the session assumes that MPI is wrapped
# around the nox call; nox will not call mpirun or srun itself.
@nox.session(python=SUPPORTED_PYTHON_VERSIONS, **_VENV_BACKEND_KWARG)
@nox.parametrize("subpackage", MODEL_SUBPACKAGE_PATHS)
@nox.parametrize("selection", MODEL_MPI_TESTS_SUBSETS)
def test_model_mpi(
    session: nox.Session, selection: ModelTestsSubset, subpackage: ModelSubpackagePath
) -> None:
    """Run MPI tests for selected icon4py model subpackages."""
    _install_session_venv(session, extras=["all"], groups=["test"])

    pytest_args = _selection_to_pytest_args(selection)
    success_codes = (
        [0] if "--collect-only" in session.posargs else [0, NO_TESTS_COLLECTED_EXIT_CODE]
    )
    with session.chdir(f"model/{subpackage}"):
        session.run(
            "pytest",
            "-sv",
            "--benchmark-disable",
            "-n0",
            "--only-mpi",
            "-k mpi_tests",
            *pytest_args,
            "tests",
            *session.posargs,
            success_codes=success_codes,
        )


@nox.session(python=SUPPORTED_PYTHON_VERSIONS, **_VENV_BACKEND_KWARG)
@nox.parametrize("selection", ["basic"])
def test_testing(session: nox.Session, selection: ModelTestsSubset) -> None:
    session.notify(f"test_model-{session.python}(selection='{selection}', subpackage='testing')")


# Bindings test sessions (includes py2fgen tool tests)
# TODO(edopao,egparedes): Change 'extras' back to 'all' once mpi4py can be compiled with hpc_sdk
@nox.session(python=SUPPORTED_PYTHON_VERSIONS, **_VENV_BACKEND_KWARG)
@nox.parametrize("selection", TOOLS_BINDINGS_TESTS_SUBSETS)
def test_tools_and_bindings(session: nox.Session, selection: ToolsBindingsTestsSubset) -> None:
    """Run tests for the Fortran bindings and integration tools."""
    _install_session_venv(
        session, extras=["fortran", "io", "testing", "profiling"], groups=["test"]
    )

    datatest_flag = "--datatest-only" if selection == "datatest" else "--datatest-skip"
    pytest_base = f"pytest -sv --benchmark-disable -n {os.environ.get('NUM_PROCESSES', 'auto')} {datatest_flag}"
    success_codes = (
        [0] if "--collect-only" in session.posargs else [0, NO_TESTS_COLLECTED_EXIT_CODE]
    )
    if selection == "unittest":
        # tools/ only has unit tests, so skip it in datatest mode
        with session.chdir("tools"):
            session.run(
                *pytest_base.split(),
                "tests",
                *session.posargs,
                success_codes=success_codes,
            )
    with session.chdir("bindings"):
        session.run(
            *pytest_base.split(),
            "tests",
            *session.posargs,
            success_codes=success_codes,
        )


# -- utils --
def _install_session_venv(
    session: nox.Session,
    *args: str | Sequence[str],
    extras: Sequence[str] = (),
    groups: Sequence[str] = (),
) -> None:
    """Install session packages using uv."""
    if _use_active_venv():
        return

    # TODO(egparedes): remove this workaround once `backend` parameter is added to sessions
    if env_extras := os.environ.get("ICON4PY_NOX_UV_CUSTOM_SESSION_EXTRAS", ""):
        extras = [*extras, *re.split(r"\W+", env_extras)]
    env = dict(os.environ.items()) | {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
    session.run_install(
        "uv",
        "sync",
        *("--python", session.python),
        "--no-dev",
        *(f"--extra={e}" for e in extras),
        *(f"--group={g}" for g in groups),
        env=env,
    )
    for item in args:
        session.run_install(
            "uv", "pip", "install", *((item,) if isinstance(item, str) else item), env=env
        )


def _selection_to_pytest_args(selection: ModelTestsSubset) -> list[str]:
    pytest_args = []

    match selection:
        case "datatest":
            pytest_args.extend(["--datatest-only"])
        case "stencils":
            pytest_args.extend(["-k", "stencil_tests"])
        case "basic":
            pytest_args.extend(
                ["--datatest-skip", "-k", "not stencil_tests and not benchmark_only"]
            )
        case _:
            raise AssertionError(f"Invalid selection: {selection}")

    return pytest_args

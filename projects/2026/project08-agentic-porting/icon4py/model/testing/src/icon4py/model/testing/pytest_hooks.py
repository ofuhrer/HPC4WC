# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import contextlib
import os
import re

import numpy as np
import pytest

from icon4py.model.common import model_backends
from icon4py.model.testing import filters


__all__ = [
    "pytest_addoption",
    "pytest_benchmark_update_json",
    "pytest_collection_modifyitems",
    "pytest_configure",
    "pytest_runtest_setup",
    "pytest_sessionfinish",
]

_TEST_LEVELS = ("any", "unit", "integration", "validation")


def pytest_configure(config):
    config.addinivalue_line("markers", "datatest: this test uses binary data")
    config.addinivalue_line(
        "markers", "with_netcdf: test uses netcdf which is an optional dependency"
    )
    config.addinivalue_line(
        "markers",
        "level(name): marks test as unit, integration, or validation tests. Validation tests are excluded by default and must be explicitly requested with --level=validation",
    )

    # Check if the --enable-mixed-precision option is set and set the environment variable accordingly
    if config.getoption("--enable-mixed-precision"):
        os.environ["FLOAT_PRECISION"] = "mixed"

    # Handle datatest options: --datatest-only  and --datatest-skip
    if m_option := config.getoption("-m", []):
        m_option = [f"({m_option})"]  # add parenthesis around original k_option just in case
    if config.getoption("--datatest-only"):
        config.option.markexpr = " and ".join(["datatest", *m_option])

    if config.getoption("--datatest-skip"):
        config.option.markexpr = " and ".join(["not datatest", *m_option])

    handle_mpi_options(config)


def pytest_addoption(parser: pytest.Parser):
    """Add custom commandline options for pytest."""
    try:
        datatest = parser.getgroup("datatest", "Options for data testing")
        datatest.addoption(
            "--datatest-skip",
            action="store_true",
            default=False,
            help="Skip all data tests",
        )
        datatest.addoption(
            "--datatest-only",
            action="store_true",
            default=False,
            help="Run only data tests",
        )
    except ValueError:
        pass
    with contextlib.suppress(ValueError):
        parser.addoption(
            "--backend",
            action="store",
            default=model_backends.DEFAULT_BACKEND,
            help="GT4Py backend to use when executing stencils. Defaults to roundtrip backend, other options include gtfn_cpu, gtfn_gpu, and embedded",
        )

    with contextlib.suppress(ValueError):
        parser.addoption(
            "--grid",
            action="store",
            help="Grid to use.",
        )

    with contextlib.suppress(ValueError):
        parser.addoption(
            "--enable-mixed-precision",
            action="store_true",
            help="Switch unit tests from double to mixed-precision",
            default=False,
        )

    with contextlib.suppress(ValueError):
        parser.addoption(
            "--level",
            action="store",
            choices=_TEST_LEVELS,
            help="Set level (unit, integration, validation) of the tests to run. Defaults to 'any', which excludes validation tests.",
            default="any",
        )
    with contextlib.suppress(ValueError):
        parser.addoption(
            "--skip-stenciltest-verification",
            action="store_true",
            help="Skip verification of `StencilTest`s against reference outputs.",
            default=False,
        )

    with contextlib.suppress(ValueError):
        parser.addoption(
            "--mpi-subcomm-size",
            action="store",
            type=int,
            default=None,
            help="Size of MPI subcommunicators for parallel test execution. "
            "Total ranks must be a multiple of this value.",
        )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Modify collected test items based on command line options."""
    scheduler = getattr(config, "_mpi_scheduler", None)
    if scheduler is not None:
        items[:] = scheduler.filter_items(items)
        if not items:
            pytest.exit("No tests assigned to this MPI subcomm group", returncode=0)

    test_level = config.getoption("--level")
    if test_level == "any":
        for item in items:
            if (marker := item.get_closest_marker("level")) is not None:
                assert all(level in _TEST_LEVELS for level in marker.args), (
                    f"Invalid test level argument on function '{item.name}' - possible values are {_TEST_LEVELS}"
                )
                if "validation" in marker.args:
                    item.add_marker(
                        pytest.mark.skip(
                            reason="Validation tests must be explicitly requested with --level=validation."
                        )
                    )
        return

    def _matches_level(item: pytest.Item) -> bool:
        if (marker := item.get_closest_marker("level")) is not None:
            assert all(level in _TEST_LEVELS for level in marker.args), (
                f"Invalid test level argument on function '{item.name}' - possible values are {_TEST_LEVELS}"
            )
            return test_level in marker.args
        return test_level == "unit"

    matched_items = []
    removed_items = []
    for item in items:
        if _matches_level(item):
            matched_items.append(item)
        else:
            removed_items.append(item)
    if removed_items:
        config.hook.pytest_deselected(items=removed_items)
    items[:] = matched_items


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Apply test item filters as the final test setup step."""

    item_marker_filters = filters.item_marker_filters
    for marker_name in set(m.name for m in item.iter_markers()) & item_marker_filters.keys():
        item_filter = item_marker_filters[marker_name]
        if item_filter.condition(item):
            item_filter.action()


_name_from_fullname_pattern = re.compile(
    r"""
        ::(?P<class>[A-Za-z_]\w*)       # capture class name
        (?::: [A-Za-z_]\w*              # skip method name
        (?:\[(?P<params>.+)\])? )       # optional parameterization; allow '[' and ']' inside params
        """,
    re.VERBOSE,
)


def _name_from_fullname(fullname: str) -> str:
    match = _name_from_fullname_pattern.search(fullname)
    if match is None:
        return fullname  # assume already fixed
    class_name = match.group("class")
    params = match.group("params")
    return f"{class_name}[{params}]" if params else class_name


# pytest benchmark hook, see:
#     https://pytest-benchmark.readthedocs.io/en/latest/hooks.html#pytest_benchmark.hookspec.pytest_benchmark_update_json
def pytest_benchmark_update_json(output_json):
    """
    Replace 'fullname' of pytest benchmarks with a shorter name for better readability in bencher.

    Note:
    Currently works only for 'StencilTest's as they have the following fixed structure:
      '<path>::<class_name>::test_stencil[<variant>]'.
    """

    for bench in output_json["benchmarks"]:
        bench["fullname"] = _name_from_fullname(bench["fullname"])
        # if GT4Py metrics are gathered, replace the benchmark stats used by `bencher` with the GT4Py metrics stats
        # to avoid reporting python overheads in `bencher` so that the results are comparable to the Fortran stencil benchmarks
        if "extra_info" in bench and "gtx_metrics" in bench["extra_info"]:
            gt4py_metrics_runtimes = bench.get("extra_info").get("gtx_metrics")
            assert len(gt4py_metrics_runtimes) > 0, (
                "No GT4Py metrics collected despite COLLECT_METRICS_LEVEL > 0"
            )
            bench["stats"]["mean"] = np.mean(gt4py_metrics_runtimes)
            bench["stats"]["median"] = np.median(gt4py_metrics_runtimes)
            bench["stats"]["stddev"] = np.std(gt4py_metrics_runtimes)
            bench["stats"]["q1"] = np.percentile(gt4py_metrics_runtimes, 25)
            bench["stats"]["q3"] = np.percentile(gt4py_metrics_runtimes, 75)
            bench["stats"]["iqr"] = bench["stats"]["q3"] - bench["stats"]["q1"]
            bench["stats"]["min"] = np.min(gt4py_metrics_runtimes)
            bench["stats"]["max"] = np.max(gt4py_metrics_runtimes)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Gather GT4Py timer metrics from benchmark fixture and add them to the test report.
    """
    outcome = yield
    report = outcome.get_result()
    if call.when == "call":
        benchmark = item.funcargs.get("benchmark", None)
        if benchmark and hasattr(benchmark, "extra_info"):
            info = benchmark.extra_info.get("gtx_metrics", None)
            if info:
                filtered_benchmark_name = benchmark.name.split("test_Test")[-1]
                # Combine the benchmark name in a readable form with the gtx_metrics data
                report.sections.append(("benchmark-extra", tuple([filtered_benchmark_name, info])))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Add a custom section to the terminal summary with GT4Py timer metrics from benchmarks.
    """
    # Gather gtx_metrics
    benchmark_gtx_metrics = []
    for outcome in ("passed", "failed", "skipped"):
        all_reports = terminalreporter.stats.get(outcome, [])
        for report in all_reports:
            for secname, info in getattr(report, "sections", []):
                if secname == "benchmark-extra":
                    benchmark_gtx_metrics.append(info)
    # Calculate the maximum length of benchmark names for formatting
    max_name_len = 0
    for benchmark_name, _ in benchmark_gtx_metrics:
        max_name_len = max(len(benchmark_name), max_name_len)
    # Print the GT4Py timer report table
    if benchmark_gtx_metrics:
        terminalreporter.ensure_newline()
        header = f"{'Benchmark Name':<{max_name_len}} | {'Mean (s)':>10} | {'Median (s)':>10} | {'Std Dev':>10} | {'Runs':>4}"
        title = " GT4Py Timer Report "
        sep_len = max(0, len(header) - len(title))
        left = sep_len // 2
        right = sep_len - left
        terminalreporter.line("-" * left + title + "-" * right, bold=True, blue=True)
        terminalreporter.line(header)
        terminalreporter.line("-" * len(header), blue=True)

        for benchmark_name, gtx_metrics in benchmark_gtx_metrics:
            terminalreporter.line(
                f"{benchmark_name:<{max_name_len}} | {np.mean(gtx_metrics):>10.8f} | {np.median(gtx_metrics):>10.8f} | {np.std(gtx_metrics):>10.8f} | {len(gtx_metrics):>4}"
            )
        terminalreporter.line("-" * len(header), blue=True)


def handle_mpi_options(config):
    with_mpi = config.getoption("--with-mpi", default=False)
    only_mpi = config.getoption("--only-mpi", default=False)
    subcomm_size = config.getoption("--mpi-subcomm-size", default=None)

    if subcomm_size is None:
        env_val = os.environ.get("ICON4PY_TEST_MPI_SUBCOMM_SIZE")
        if env_val is not None:
            subcomm_size = int(env_val)
    if subcomm_size is not None and not (with_mpi or only_mpi):
        raise pytest.UsageError(
            "--mpi-subcomm-size requires MPI to be initialized. Make sure --with-mpi or --only-mpi is passed."
        )

    if with_mpi or only_mpi:
        from icon4py.model.common.decomposition.mpi_decomposition import (  # noqa: PLC0415 [import-outside-top-level]
            import_error,
            mpi4py,
        )

        if mpi4py is None:
            raise pytest.UsageError(
                f"--with-mpi requires mpi4py and ghex, but import failed: {import_error}"
            )

        from icon4py.model.common.decomposition.mpi_decomposition import (  # noqa: PLC0415 [import-outside-top-level]
            init_mpi,
        )

        init_mpi()

        if subcomm_size is not None:
            scheduler = MPISubcommScheduler(subcomm_size)
            config._mpi_scheduler = scheduler

            if scheduler.subcomm.Get_rank() == 0:
                start_rank = scheduler.group_id * scheduler.subcomm_size
                end_rank = (scheduler.group_id + 1) * scheduler.subcomm_size - 1
                print(
                    f"\n[MPI Scheduler] Group {scheduler.group_id}/{scheduler.num_groups}: "
                    f"world ranks {start_rank}-{end_rank}, subcomm size {scheduler.subcomm_size}"
                )


class MPISubcommScheduler:
    """Splits MPI_COMM_WORLD into subcommunicators for parallel test execution."""

    def __init__(self, subcomm_size: int):
        from mpi4py import MPI  # noqa: PLC0415 [import-outside-top-level]

        if subcomm_size <= 0:
            raise ValueError("--mpi-subcomm-size must be a positive integer")

        self.world = MPI.COMM_WORLD
        self.world_size = self.world.Get_size()
        self.world_rank = self.world.Get_rank()
        self.subcomm_size = subcomm_size
        self._finalized = False

        if self.world_size % self.subcomm_size != 0:
            raise ValueError(
                f"MPI world size ({self.world_size}) must be divisible by "
                f"--mpi-subcomm-size ({self.subcomm_size})"
            )

        self.num_groups = self.world_size // self.subcomm_size
        self.group_id = self.world_rank // self.subcomm_size
        self.subcomm = self.world.Split(self.group_id, self.world_rank)

        from icon4py.model.common.decomposition import (  # noqa: PLC0415 [import-outside-top-level]
            mpi_decomposition,
        )

        self._original_get_props = mpi_decomposition._get_process_properties

        def _patched_get_props(with_mpi=False, comm_id=None, **kwargs):
            if with_mpi and comm_id is None:
                comm_id = self.subcomm
            return self._original_get_props(with_mpi=with_mpi, comm_id=comm_id, **kwargs)

        mpi_decomposition._get_process_properties = _patched_get_props

    def filter_items(self, items: list[pytest.Item]) -> list[pytest.Item]:
        """Return only the test items assigned to this subcomm group."""
        mpi_items = sorted(
            (i for i in items if i.get_closest_marker("mpi")),
            key=lambda i: i.nodeid,
        )
        non_mpi_items = [i for i in items if i not in mpi_items]

        valid_mpi_items = [
            item
            for item in mpi_items
            if item.get_closest_marker("mpi").kwargs.get("min_size", 1) <= self.subcomm_size
        ]

        assigned_mpi = [
            item
            for idx, item in enumerate(valid_mpi_items)
            if idx % self.num_groups == self.group_id
        ]

        if self.group_id == 0:
            return non_mpi_items + assigned_mpi
        return assigned_mpi

    def finalize(self) -> None:
        """Free the subcommunicator and restore patched functions."""
        if self._finalized:
            return
        self._finalized = True

        try:
            if self.subcomm is not None and self.subcomm != self.world:
                self.subcomm.Free()
        finally:
            from icon4py.model.common.decomposition import (  # noqa: PLC0415 [import-outside-top-level]
                mpi_decomposition,
            )

            mpi_decomposition._get_process_properties = self._original_get_props


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    scheduler = getattr(session.config, "_mpi_scheduler", None)
    if scheduler is not None:
        scheduler.finalize()

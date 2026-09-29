from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import numpy as np

from benchmarks.common import (
    BenchmarkResult,
    WorkloadResult,
    arrays_close,
    compile_cpp_executable,
    load_workload,
    numpy_reference,
    print_csv,
    run_key_value_executable,
    validate_arguments,
    workload_directory,
    write_csv,
)

Implementation = Literal[
    "yasmin_numpy",
    "yasmin_cpp",
    "yasmin_openmp",
    "numpy",
    "cpp",
    "cpp_openmp",
    "gt4py_cpu",
    "gt4py_numpy",
]

DEFAULT_SIZES = [128, 256, 512, 1024, 2048]
DEFAULT_IMPLEMENTATIONS: list[Implementation] = [
    "yasmin_numpy",
    "numpy",
    "yasmin_cpp",
    "cpp",
]
OPENMP_IMPLEMENTATIONS: tuple[Implementation, Implementation] = (
    "yasmin_openmp",
    "cpp_openmp",
)
DEFAULT_OUTPUT_DIR = Path("benchmarks/outputs/raw/csv")


def run(
    *,
    workload: str,
    sizes: list[int],
    implementations: list[Implementation],
    warmups: int,
    repeats: int,
    cxx: str,
    threads: int | None,
    gt4py_cpu_thread_counts: list[int] | None = None,
) -> dict[int, list[BenchmarkResult]]:
    results_by_size: dict[int, list[BenchmarkResult]] = {}

    for nx in sizes:
        results: list[BenchmarkResult] = []

        for implementation in implementations:
            implementation_thread_counts = (
                sorted(set(gt4py_cpu_thread_counts))
                if implementation == "gt4py_cpu" and gt4py_cpu_thread_counts is not None
                else [threads]
            )

            for implementation_threads in implementation_thread_counts:
                results.append(
                    _run_implementation(
                        workload=workload,
                        implementation=implementation,
                        nx=nx,
                        ny=nx,
                        warmups=warmups,
                        repeats=repeats,
                        cxx=cxx,
                        threads=implementation_threads,
                    )
                )

        results_by_size[nx] = results

    return results_by_size


def run_scaling(
    *,
    workload: str,
    nx: int,
    thread_counts: list[int],
    warmups: int,
    repeats: int,
    cxx: str,
    mode: Literal["strong", "weak"] = "strong",
    include_gt4py: bool = False,
) -> list[BenchmarkResult]:
    if not thread_counts or 1 not in thread_counts or any(t < 1 for t in thread_counts):
        raise ValueError("Thread counts must be positive and include 1")

    implementations = list(OPENMP_IMPLEMENTATIONS)
    if include_gt4py:
        implementations.append("gt4py_cpu")

    return [
        _run_implementation(
            workload=workload,
            implementation=implementation,
            nx=nx,
            ny=nx * threads if mode == "weak" else nx,
            warmups=warmups,
            repeats=repeats,
            cxx=cxx,
            threads=threads,
        )
        for threads in sorted(set(thread_counts))
        for implementation in implementations
    ]


def _run_implementation(
    *,
    workload: str,
    implementation: Implementation,
    nx: int,
    ny: int,
    warmups: int,
    repeats: int,
    cxx: str,
    threads: int | None,
) -> BenchmarkResult:
    validate_arguments(nx, ny, warmups, repeats)
    if threads is not None and threads < 1:
        raise ValueError("Expected threads >= 1")

    if implementation in ("cpp", "cpp_openmp"):
        result = run_cpp(
            workload=workload,
            backend="openmp" if implementation == "cpp_openmp" else "cpp",
            nx=nx,
            ny=ny,
            warmups=warmups,
            repeats=repeats,
            cxx=cxx,
            threads=threads,
        )
    else:
        result = run_python(
            workload=workload,
            implementation=implementation,
            nx=nx,
            ny=ny,
            warmups=warmups,
            repeats=repeats,
            cxx=cxx,
            threads=threads,
        )

    expected = numpy_reference(workload, nx, ny)
    correct = result.output.size == expected.size and arrays_close(
        result.output.reshape(expected.shape),
        expected,
    )

    return BenchmarkResult(
        implementation=implementation,
        nx=nx,
        ny=ny,
        threads=(
            (threads or 1)
            if implementation == "gt4py_cpu"
            else (threads if implementation in OPENMP_IMPLEMENTATIONS else None)
        ),
        runtime_ms=result.runtime_ms,
        correct=correct,
    )


def run_cpp(
    *,
    workload: str,
    backend: Literal["cpp", "openmp"],
    nx: int,
    ny: int,
    warmups: int,
    repeats: int,
    cxx: str,
    threads: int | None,
) -> WorkloadResult:
    source = workload_directory(workload) / f"{workload}_{backend}.cpp"
    executable = compile_cpp_executable(source, cxx=cxx, openmp=backend == "openmp")
    env = os.environ.copy()

    if backend == "openmp":
        env["OMP_DYNAMIC"] = "FALSE"
        if threads is not None:
            env["OMP_NUM_THREADS"] = str(threads)

    try:
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / f"{workload}.bin"
            output = run_key_value_executable(
                executable.path,
                [str(nx), str(ny), str(warmups), str(repeats), str(output_path)],
                env=env,
            )
            actual = np.fromfile(output_path, dtype=np.float64)
    finally:
        executable.directory.cleanup()

    return WorkloadResult(
        output=actual,
        runtime_ms=float(output["RUNTIME_MS"]),
    )


def run_python(
    *,
    workload: str,
    implementation: Implementation,
    nx: int,
    ny: int,
    warmups: int,
    repeats: int,
    cxx: str,
    threads: int | None,
) -> WorkloadResult:
    result: WorkloadResult

    if implementation in ("gt4py_cpu", "gt4py_numpy"):
        workload_directory(workload)
        env = os.environ.copy()
        env["CXX"] = cxx
        env["OMP_NUM_THREADS"] = (
            str(threads or 1) if implementation == "gt4py_cpu" else "1"
        )
        env["OMP_DYNAMIC"] = "FALSE"

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / f"{workload}.bin"
            command = [
                "-m",
                f"benchmarks.{workload}.{workload}_gt4py",
                "--backend",
                "gt:cpu_ifirst" if implementation == "gt4py_cpu" else "numpy",
                "--nx",
                str(nx),
                "--ny",
                str(ny),
                "--warmups",
                str(warmups),
                "--repeats",
                str(repeats),
                "--output",
                str(output_path),
            ]

            try:
                values = run_key_value_executable(
                    Path(sys.executable),
                    command,
                    env=env,
                )
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f"GT4Py run failed. Check the compiler/environment.\n{error.stderr}"
                ) from error

            result = WorkloadResult(
                output=np.fromfile(output_path, dtype=np.float64),
                runtime_ms=float(values["RUNTIME_MS"]),
            )

    elif implementation == "numpy":
        module = load_workload(workload, "numpy")
        result = module.main(
            nx=nx,
            ny=ny,
            warmups=warmups,
            repeats=repeats,
        )

    else:
        module = load_workload(workload, "yasmin")

        get_affinity = cast(
            Callable[[int], set[int]] | None,
            getattr(os, "sched_getaffinity", None),
        )
        set_affinity = cast(
            Callable[[int, set[int]], None] | None,
            getattr(os, "sched_setaffinity", None),
        )

        original_affinity = get_affinity(0) if get_affinity is not None else None

        try:
            result = module.main(
                backend=implementation.removeprefix("yasmin_"),
                nx=nx,
                ny=ny,
                warmups=warmups,
                repeats=repeats,
                cxx=cxx,
                threads=threads,
            )
        finally:
            if original_affinity is not None and set_affinity is not None:
                set_affinity(0, original_affinity)

    return result


def _selected_implementations(
    implementations: list[Implementation] | None,
    *,
    include_openmp: bool,
    include_gt4py: bool = False,
) -> list[Implementation]:
    selected = list(implementations or DEFAULT_IMPLEMENTATIONS)

    if include_openmp:
        for implementation in OPENMP_IMPLEMENTATIONS:
            if implementation not in selected:
                selected.append(implementation)

    if include_gt4py:
        for implementation in ("gt4py_cpu", "gt4py_numpy"):
            if implementation not in selected:
                selected.append(implementation)

    return selected


def _write_outputs(
    *,
    workload: str,
    output_dir: Path,
    results_by_size: dict[int, list[BenchmarkResult]],
) -> None:
    for nx, results in results_by_size.items():
        write_csv(output_dir / f"{workload}_{nx}.csv", results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Yasmin benchmarks.")
    parser.add_argument("--workload", default="laplacian")
    parser.add_argument(
        "--size",
        type=int,
        action="append",
        dest="sizes",
        help="Grid side length; weak scaling uses this as the one-thread size.",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--gt4py-cpu-thread-counts",
        type=int,
        nargs="+",
        help="Thread counts for GT4Py CPU during runtime size sweeps.",
    )

    scaling = parser.add_mutually_exclusive_group()
    scaling.add_argument("--strong-scaling", action="store_true")
    scaling.add_argument("--weak-scaling", action="store_true")

    parser.add_argument("--thread-counts", type=int, nargs="+")
    parser.add_argument(
        "--implementation",
        choices=(
            "yasmin_numpy",
            "yasmin_cpp",
            "yasmin_openmp",
            "numpy",
            "cpp",
            "cpp_openmp",
            "gt4py_cpu",
            "gt4py_numpy",
        ),
        action="append",
        dest="implementations",
        help="Implementation to benchmark. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--include-openmp",
        action="store_true",
        help="Include Yasmin OpenMP and standalone OpenMP implementations.",
    )
    parser.add_argument(
        "--include-gt4py",
        action="store_true",
        help="Include GT4Py CPU and NumPy backends; scaling includes CPU only.",
    )
    parser.add_argument(
        "--cxx",
        default=os.environ.get("CXX") or "c++",
        help="C++ compiler used for native and standalone C++ benchmarks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where per-size CSV files are written.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Print CSV rows without writing benchmark output files.",
    )

    args = parser.parse_args()

    if args.strong_scaling or args.weak_scaling:
        mode: Literal["strong", "weak"] = "weak" if args.weak_scaling else "strong"

        if args.sizes and len(args.sizes) != 1:
            parser.error("Scaling requires exactly one --size")

        if (
            args.threads is not None
            or args.gt4py_cpu_thread_counts is not None
            or args.implementations
            or args.include_openmp
        ):
            parser.error(
                "Scaling selects its own implementations and thread counts; "
                "use --thread-counts and optionally --include-gt4py"
            )

        thread_counts = args.thread_counts or [1, 2, 4]
        if 1 not in thread_counts or any(t < 1 for t in thread_counts):
            parser.error("Thread counts must be positive and include 1")

        results = run_scaling(
            workload=args.workload,
            nx=(args.sizes or [512 if args.weak_scaling else 2048])[0],
            thread_counts=thread_counts,
            warmups=args.warmups,
            repeats=args.repeats,
            cxx=args.cxx,
            mode=mode,
            include_gt4py=args.include_gt4py,
        )

        print_csv(results)

        if not args.no_output:
            write_csv(
                args.output_dir / f"{args.workload}_{mode}_scaling.csv",
                results,
            )

        return

    if args.thread_counts is not None:
        parser.error("--thread-counts requires --strong-scaling or --weak-scaling")

    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")

    if args.gt4py_cpu_thread_counts is not None and any(
        threads < 1 for threads in args.gt4py_cpu_thread_counts
    ):
        parser.error("GT4Py CPU thread counts must be positive")

    implementations = _selected_implementations(
        args.implementations,
        include_openmp=args.include_openmp,
        include_gt4py=args.include_gt4py,
    )

    results_by_size = run(
        workload=args.workload,
        sizes=args.sizes or DEFAULT_SIZES,
        implementations=implementations,
        warmups=args.warmups,
        repeats=args.repeats,
        cxx=args.cxx,
        threads=args.threads,
        gt4py_cpu_thread_counts=args.gt4py_cpu_thread_counts,
    )

    all_results = [result for results in results_by_size.values() for result in results]
    print_csv(all_results)

    if not args.no_output:
        _write_outputs(
            workload=args.workload,
            output_dir=args.output_dir,
            results_by_size=results_by_size,
        )


if __name__ == "__main__":
    main()

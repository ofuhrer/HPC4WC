from __future__ import annotations

import argparse
import csv
import importlib
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]
TimedCallable = Callable[[], None]


def workload_directory(workload: str) -> Path:
    if not workload.isidentifier():
        raise ValueError(f"Invalid workload name: {workload!r}")
    directory = Path(__file__).resolve().parent / workload
    if not (directory / f"{workload}_numpy.py").is_file():
        raise ValueError(f"Unknown workload: {workload!r}")
    return directory


def load_workload(workload: str, implementation: str) -> ModuleType:
    workload_directory(workload)
    return importlib.import_module(f"benchmarks.{workload}.{workload}_{implementation}")


def numpy_reference(
    workload: str,
    nx: int,
    ny: int,
) -> Array:
    module = load_workload(workload, "numpy")
    expected: Array = module.reference(module.make_initial(nx, ny))
    return expected


@dataclass(frozen=True, slots=True)
class WorkloadResult:
    output: Array
    runtime_ms: float


def workload_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("laplacian.bin"))
    return parser


def validate_arguments(
    nx: int,
    ny: int,
    warmups: int,
    repeats: int,
) -> None:
    if nx < 2 or ny < 2 or warmups < 0 or repeats < 1:
        raise ValueError("Expected nx >= 2, ny >= 2, warmups >= 0, repeats >= 1")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    implementation: str
    nx: int
    ny: int
    threads: int | None
    runtime_ms: float
    correct: bool

    def as_row(self) -> dict[str, str]:
        return {
            "implementation": self.implementation,
            "nx": str(self.nx),
            "ny": str(self.ny),
            "threads": "" if self.threads is None else str(self.threads),
            "runtime_ms": f"{self.runtime_ms:.6f}",
            "correct": str(self.correct).lower(),
        }


@dataclass(frozen=True, slots=True)
class CompiledExecutable:
    path: Path
    directory: tempfile.TemporaryDirectory[str]


def median_runtime_ms(
    action: TimedCallable,
    *,
    warmups: int,
    repeats: int,
) -> float:
    for _ in range(warmups):
        action()

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        action()
        timings.append((time.perf_counter() - start) * 1_000)

    return statistics.median(timings)


def arrays_close(actual: Array, expected: Array) -> bool:
    return bool(np.allclose(actual, expected, rtol=1e-12, atol=1e-12))


def compile_cpp_executable(
    source: Path,
    *,
    cxx: str,
    openmp: bool,
    extra_flags: Sequence[str] = (),
) -> CompiledExecutable:
    directory = tempfile.TemporaryDirectory()
    executable = Path(directory.name) / source.stem

    command = [
        cxx,
        "-O3",
        "-std=c++17",
    ]

    if openmp:
        command.append("-fopenmp")

    command.extend(extra_flags)

    command.extend(
        [
            str(source),
            "-o",
            str(executable),
        ]
    )

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        directory.cleanup()
        raise RuntimeError(
            f"Failed to compile {source.name} with {cxx}:\n{error.stderr}"
        ) from error
    except OSError as error:
        directory.cleanup()
        raise RuntimeError(f"Failed to run compiler {cxx!r}: {error}") from error

    return CompiledExecutable(
        path=executable,
        directory=directory,
    )


def run_key_value_executable(
    executable: Path,
    args: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    completed = subprocess.run(
        [str(executable), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", maxsplit=1)
        values[key.strip()] = value.strip()

    return values


def print_csv(results: Sequence[BenchmarkResult]) -> None:
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "implementation",
            "nx",
            "ny",
            "threads",
            "runtime_ms",
            "correct",
        ],
    )
    writer.writeheader()
    writer.writerows(result.as_row() for result in results)


def write_csv(path: Path, results: Sequence[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "implementation",
                "nx",
                "ny",
                "threads",
                "runtime_ms",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(result.as_row() for result in results)

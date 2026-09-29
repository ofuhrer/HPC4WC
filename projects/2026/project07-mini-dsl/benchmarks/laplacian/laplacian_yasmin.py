from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

import yasmin as yasi
from benchmarks.common import (
    WorkloadResult,
    median_runtime_ms,
    validate_arguments,
    workload_parser,
)
from benchmarks.laplacian.laplacian_numpy import make_initial
from yasmin.frontend.expr import SymbolicExpr


@dataclass(frozen=True, slots=True)
class LaplacianWorkload:
    operator: yasi.Operator
    u: yasi.Field
    out: yasi.Field


def laplacian() -> LaplacianWorkload:
    x = yasi.Dimension("x")
    y = yasi.Dimension("y")

    u = yasi.Field("u", dims=(x, y), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x, y), dtype=yasi.float64)

    @yasi.stencil
    def laplace(field: yasi.Field) -> SymbolicExpr:
        return (
            field[-1, 0] + field[1, 0] + field[0, -1] + field[0, 1] - 4.0 * field[0, 0]
        )

    @yasi.operator
    def apply_laplacian(
        input_field: yasi.Field,
        output_field: yasi.Field,
    ) -> None:
        output_field[0, 0] = laplace(input_field)

    return LaplacianWorkload(
        operator=apply_laplacian(u, out),
        u=u,
        out=out,
    )


def main(
    *,
    nx: int,
    ny: int,
    warmups: int,
    repeats: int,
    backend: Literal["numpy", "cpp", "openmp"] = "numpy",
    cxx: str = "c++",
    threads: int | None = None,
    output: Path | None = None,
) -> WorkloadResult:
    validate_arguments(nx, ny, warmups, repeats)

    if threads is not None and threads < 1:
        raise ValueError("Expected threads >= 1")

    workload = laplacian()

    u = make_initial(nx, ny)
    out = np.zeros_like(u)

    fields = {
        workload.u: u,
        workload.out: out,
    }

    if backend == "numpy":

        def execute_once() -> None:
            yasi.execute(
                workload.operator,
                backend="numpy",
                fields=fields,
            )

    else:

        def compile_kernel() -> yasi.Kernel:
            if backend == "openmp":
                return yasi.compile(
                    workload.operator,
                    config=yasi.CompileConfig(
                        backend="openmp",
                        options=yasi.OpenMPOptions(
                            use_collapse=True,
                            schedule="static",
                            num_threads=threads,
                            adaptive=False,
                            extra_compile_flags=(),
                        ),
                    ),
                )

            return yasi.compile(
                workload.operator,
                backend="cpp",
            )

        previous_cxx = os.environ.get("CXX")
        os.environ["CXX"] = cxx

        try:
            kernel = compile_kernel()
        finally:
            if previous_cxx is None:
                os.environ.pop("CXX", None)
            else:
                os.environ["CXX"] = previous_cxx

        def execute_once() -> None:
            kernel(fields=fields)

    runtime_ms = median_runtime_ms(
        execute_once,
        warmups=warmups,
        repeats=repeats,
    )

    if output is not None:
        out.tofile(output)

    return WorkloadResult(
        output=out,
        runtime_ms=runtime_ms,
    )


if __name__ == "__main__":
    parser = workload_parser()
    parser.add_argument(
        "--backend", choices=("numpy", "cpp", "openmp"), default="numpy"
    )
    parser.add_argument("--cxx", default=os.environ.get("CXX") or "c++")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    result = main(**vars(args))
    print(f"NX={args.nx}\nNY={args.ny}\nRUNTIME_MS={result.runtime_ms:.12f}")

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yasmin as yasi
from benchmarks.advection_diffusion.advection_diffusion_numpy import make_initial
from benchmarks.common import (
    WorkloadResult,
    median_runtime_ms,
    validate_arguments,
    workload_parser,
)
from yasmin.frontend.expr import SymbolicExpr


def advection_diffusion(
    nx: int,
) -> tuple[yasi.Operator, tuple[yasi.Field, ...], yasi.Scalar]:
    x, y = yasi.Dimension("x"), yasi.Dimension("y")
    fields = tuple(
        yasi.Field(name, dims=(x, y), dtype=yasi.float64)
        for name in ("u", "v", "out_u", "out_v")
    )
    u, v, out_u, out_v = fields
    dt = yasi.Scalar("dt", dtype=yasi.float64)
    dx = 1.0 / (nx - 1)

    @yasi.stencil
    def update(f: yasi.Field, u: yasi.Field, v: yasi.Field) -> SymbolicExpr:
        # Fixed positive input velocities make absolute-value fields unnecessary.
        adv_x = u[0, 0] / (60.0 * dx) * (
            45.0 * (f[1, 0] - f[-1, 0])
            - 9.0 * (f[2, 0] - f[-2, 0])
            + (f[3, 0] - f[-3, 0])
        ) - u[0, 0] / (60.0 * dx) * (
            f[3, 0]
            + f[-3, 0]
            - 6.0 * (f[2, 0] + f[-2, 0])
            + 15.0 * (f[1, 0] + f[-1, 0])
            - 20.0 * f[0, 0]
        )

        adv_y = v[0, 0] / (60.0 * dx) * (
            45.0 * (f[0, 1] - f[0, -1])
            - 9.0 * (f[0, 2] - f[0, -2])
            + (f[0, 3] - f[0, -3])
        ) - v[0, 0] / (60.0 * dx) * (
            f[0, 3]
            + f[0, -3]
            - 6.0 * (f[0, 2] + f[0, -2])
            + 15.0 * (f[0, 1] + f[0, -1])
            - 20.0 * f[0, 0]
        )

        diff_x = (
            0.0 - f[-2, 0] + 16.0 * f[-1, 0] - 30.0 * f[0, 0] + 16.0 * f[1, 0] - f[2, 0]
        ) / (12.0 * dx * dx)

        diff_y = (
            0.0 - f[0, -2] + 16.0 * f[0, -1] - 30.0 * f[0, 0] + 16.0 * f[0, 1] - f[0, 2]
        ) / (12.0 * dx * dx)

        return f[0, 0] + dt * (0.0 - (adv_x + adv_y) + 0.1 * (diff_x + diff_y))

    @yasi.operator
    def stage(
        u: yasi.Field,
        v: yasi.Field,
        out_u: yasi.Field,
        out_v: yasi.Field,
    ) -> None:
        out_u[0, 0] = update(u, u, v)
        out_v[0, 0] = update(v, u, v)

    return stage(u, v, out_u, out_v), fields, dt


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

    initial = make_initial(nx, ny)
    out = initial.copy()

    operator, fields, dt_scalar = advection_diffusion(nx)
    u, v, out_u, out_v = fields

    kernel = None

    if backend != "numpy":
        previous_cxx = os.environ.get("CXX")
        os.environ["CXX"] = cxx

        try:
            if backend == "openmp":
                kernel = yasi.compile(
                    operator,
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
            else:
                kernel = yasi.compile(operator, backend="cpp")
        finally:
            if previous_cxx is None:
                os.environ.pop("CXX", None)
            else:
                os.environ["CXX"] = previous_cxx

    dx = 1.0 / (nx - 1)

    scalars = {
        dt_scalar: dx * dx,
    }

    bindings = {
        u: initial[0],
        v: initial[1],
        out_u: out[0],
        out_v: out[1],
    }

    def execute_once() -> None:
        if kernel is None:
            yasi.execute(
                operator,
                backend="numpy",
                fields=bindings,
                scalars=scalars,
            )
        else:
            kernel(
                fields=bindings,
                scalars=scalars,
            )

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
    parser.set_defaults(output=Path("advection_diffusion.bin"))
    parser.add_argument(
        "--backend",
        choices=("numpy", "cpp", "openmp"),
        default="numpy",
    )
    parser.add_argument("--cxx", default=os.environ.get("CXX") or "c++")
    parser.add_argument("--threads", type=int)

    args = parser.parse_args()
    result = main(**vars(args))

    print(f"NX={args.nx}\nNY={args.ny}\nRUNTIME_MS={result.runtime_ms:.12f}")

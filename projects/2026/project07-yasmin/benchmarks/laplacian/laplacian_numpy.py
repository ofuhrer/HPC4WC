from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt

from benchmarks.common import (
    WorkloadResult,
    median_runtime_ms,
    validate_arguments,
    workload_parser,
)


def make_initial(
    nx: int,
    ny: int,
) -> npt.NDArray[np.float64]:
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)

    values_x = np.sin(2.0 * math.pi * x / (nx - 1))
    values_y = np.sin(2.0 * math.pi * y / (ny - 1))

    return np.outer(values_x, values_y).astype(np.float64)


def reference(
    u: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    out = np.zeros_like(u)
    laplacian(u, out)
    return out


def laplacian(
    u: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
) -> None:
    out[1:-1, 1:-1] = (
        u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4.0 * u[1:-1, 1:-1]
    )


def main(
    *,
    nx: int,
    ny: int,
    warmups: int,
    repeats: int,
    output: Path | None = None,
) -> WorkloadResult:
    validate_arguments(nx, ny, warmups, repeats)

    u = make_initial(nx, ny)
    out = np.zeros_like(u)

    runtime_ms = median_runtime_ms(
        lambda: laplacian(u, out),
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
    args = workload_parser().parse_args()
    result = main(**vars(args))
    print(f"NX={args.nx}\nRUNTIME_MS={result.runtime_ms:.12f}")

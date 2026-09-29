from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import gt4py.cartesian.gtscript as gtscript
import gt4py.storage as storage
import numpy as np
from gt4py.cartesian.gtscript import PARALLEL, computation, interval

from benchmarks.common import (
    WorkloadResult,
    median_runtime_ms,
    validate_arguments,
    workload_parser,
)
from benchmarks.laplacian.laplacian_numpy import make_initial

if TYPE_CHECKING:
    FloatField: TypeAlias = Any
else:
    FloatField = gtscript.Field[np.float64]


def laplacian(u: FloatField, out: FloatField) -> None:
    with computation(PARALLEL), interval(...):
        # GTScript assignment writes an output field.
        out = (  # noqa: F841
            u[-1, 0, 0] + u[1, 0, 0] + u[0, -1, 0] + u[0, 1, 0] - 4.0 * u[0, 0, 0]
        )


def main(
    *,
    nx: int,
    ny: int,
    warmups: int,
    repeats: int,
    output: Path | None = None,
    backend: Literal["gt:cpu_ifirst", "numpy"] = "gt:cpu_ifirst",
) -> WorkloadResult:
    validate_arguments(nx, ny, warmups, repeats)

    origin = (1, 1, 0)
    initial = make_initial(nx, ny)

    u = storage.from_array(
        initial[:, :, None],
        np.float64,
        backend=backend,
        aligned_index=origin,
    )

    out = storage.zeros(
        (nx, ny, 1),
        np.float64,
        backend=backend,
        aligned_index=origin,
    )

    kernel = gtscript.stencil(
        backend=backend,
        definition=laplacian,
    )

    def execute_once() -> None:
        if nx > 2 and ny > 2:
            kernel(
                u=u,
                out=out,
                origin=origin,
                domain=(nx - 2, ny - 2, 1),
            )

    execute_once()

    runtime_ms = median_runtime_ms(
        execute_once,
        warmups=warmups,
        repeats=repeats,
    )

    result = np.array(out[:, :, 0], copy=True)

    if output is not None:
        result.tofile(output)

    return WorkloadResult(
        output=result,
        runtime_ms=runtime_ms,
    )


if __name__ == "__main__":
    parser = workload_parser()
    parser.add_argument(
        "--backend", choices=("gt:cpu_ifirst", "numpy"), default="gt:cpu_ifirst"
    )
    args = parser.parse_args()
    result = main(**vars(args))
    print(f"NX={args.nx}\nNY={args.ny}\nRUNTIME_MS={result.runtime_ms:.12f}")

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from benchmarks.common import (
    WorkloadResult,
    median_runtime_ms,
    validate_arguments,
    workload_parser,
)

Array = npt.NDArray[np.float64]
MU = 0.1


def solution(t: float, x: Array, y: Array) -> Array:
    delta = 1.0 / (4.0 * (1.0 + np.exp(-t - 4.0 * x + 4.0 * y) / (32.0 * MU)))
    return np.stack((0.75 - delta, 0.75 + delta))


def make_initial(nx: int, ny: int) -> Array:
    if nx < 7 or ny < 7:
        raise ValueError("Advection-diffusion requires nx >= 7 and ny >= 7")

    dx = 1.0 / (nx - 1)
    x = np.arange(nx, dtype=np.float64) * dx
    y = np.arange(ny, dtype=np.float64) * dx

    return solution(0.0, x[:, None], y[None, :])


def advection_diffusion(state: Array, out: Array) -> None:
    # The fixed initial velocities are positive, so abs(u) = u and abs(v) = v.
    dx = 1.0 / (state.shape[1] - 1)
    dy = dx
    dt = dx * dx
    u, v = state[:, 3:-3, 3:-3]
    for field, target in zip(state, out, strict=True):
        c = field[3:-3, 3:-3]
        xp1, xm1 = field[4:-2, 3:-3], field[2:-4, 3:-3]
        xp2, xm2 = field[5:-1, 3:-3], field[1:-5, 3:-3]
        xp3, xm3 = field[6:, 3:-3], field[:-6, 3:-3]
        yp1, ym1 = field[3:-3, 4:-2], field[3:-3, 2:-4]
        yp2, ym2 = field[3:-3, 5:-1], field[3:-3, 1:-5]
        yp3, ym3 = field[3:-3, 6:], field[3:-3, :-6]
        adv_x = u / (60.0 * dx) * (
            45.0 * (xp1 - xm1) - 9.0 * (xp2 - xm2) + (xp3 - xm3)
        ) - u / (60.0 * dx) * (
            xp3 + xm3 - 6.0 * (xp2 + xm2) + 15.0 * (xp1 + xm1) - 20.0 * c
        )
        adv_y = v / (60.0 * dy) * (
            45.0 * (yp1 - ym1) - 9.0 * (yp2 - ym2) + (yp3 - ym3)
        ) - v / (60.0 * dy) * (
            yp3 + ym3 - 6.0 * (yp2 + ym2) + 15.0 * (yp1 + ym1) - 20.0 * c
        )
        diff_x = (-xm2 + 16.0 * xm1 - 30.0 * c + 16.0 * xp1 - xp2) / (12.0 * dx * dx)
        diff_y = (-ym2 + 16.0 * ym1 - 30.0 * c + 16.0 * yp1 - yp2) / (12.0 * dy * dy)
        target[3:-3, 3:-3] = c + dt * (-(adv_x + adv_y) + MU * (diff_x + diff_y))


def reference(initial: Array) -> Array:
    out = initial.copy()
    advection_diffusion(initial, out)
    return out


def main(
    *, nx: int, ny: int, warmups: int, repeats: int, output: Path | None = None
) -> WorkloadResult:
    validate_arguments(nx, ny, warmups, repeats)
    initial = make_initial(nx, ny)
    out = initial.copy()
    runtime_ms = median_runtime_ms(
        lambda: advection_diffusion(initial, out),
        warmups=warmups,
        repeats=repeats,
    )
    if output is not None:
        out.tofile(output)
    return WorkloadResult(output=out, runtime_ms=runtime_ms)


if __name__ == "__main__":
    parser = workload_parser()
    parser.set_defaults(output=Path("advection_diffusion.bin"))
    args = parser.parse_args()
    result = main(**vars(args))
    print(f"NX={args.nx}\nNY={args.ny}\nRUNTIME_MS={result.runtime_ms:.12f}")

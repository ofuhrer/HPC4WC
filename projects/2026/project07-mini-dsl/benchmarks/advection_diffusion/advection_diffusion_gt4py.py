# GTScript functions below retain the notebook's DSL annotations and signatures.
# mypy: disable-error-code="no-untyped-def,untyped-decorator,valid-type,no-untyped-call"

from pathlib import Path
from typing import Any, Literal

import gt4py.cartesian.gtscript as gtscript
import gt4py.storage
import numpy as np
from gt4py.cartesian.gtscript import PARALLEL, computation, interval

from benchmarks.advection_diffusion.advection_diffusion_numpy import make_initial
from benchmarks.common import (
    WorkloadResult,
    median_runtime_ms,
    validate_arguments,
    workload_parser,
)

# Adapted from examples/cartesian/demo_burgers.ipynb in GridTools/gt4py.
# Fixed positive velocities allow u/v to replace their absolute values.


@gtscript.function
def advection_x(dx, u, abs_u, phi):
    adv_phi_x = u[0, 0, 0] / (60.0 * dx) * (
        +45.0 * (phi[1, 0, 0] - phi[-1, 0, 0])
        - 9.0 * (phi[2, 0, 0] - phi[-2, 0, 0])
        + (phi[3, 0, 0] - phi[-3, 0, 0])
    ) - abs_u[0, 0, 0] / (60.0 * dx) * (
        +(phi[3, 0, 0] + phi[-3, 0, 0])
        - 6.0 * (phi[2, 0, 0] + phi[-2, 0, 0])
        + 15.0 * (phi[1, 0, 0] + phi[-1, 0, 0])
        - 20.0 * phi[0, 0, 0]
    )
    return adv_phi_x


@gtscript.function
def advection_y(dy, v, abs_v, phi):
    adv_phi_y = v[0, 0, 0] / (60.0 * dy) * (
        +45.0 * (phi[0, 1, 0] - phi[0, -1, 0])
        - 9.0 * (phi[0, 2, 0] - phi[0, -2, 0])
        + (phi[0, 3, 0] - phi[0, -3, 0])
    ) - abs_v[0, 0, 0] / (60.0 * dy) * (
        +(phi[0, 3, 0] + phi[0, -3, 0])
        - 6.0 * (phi[0, 2, 0] + phi[0, -2, 0])
        + 15.0 * (phi[0, 1, 0] + phi[0, -1, 0])
        - 20.0 * phi[0, 0, 0]
    )
    return adv_phi_y


@gtscript.function
def advection(dx, dy, u, v):
    adv_u_x = advection_x(dx=dx, u=u, abs_u=u, phi=u)
    adv_u_y = advection_y(dy=dy, v=v, abs_v=v, phi=u)
    adv_u = adv_u_x[0, 0, 0] + adv_u_y[0, 0, 0]

    adv_v_x = advection_x(dx=dx, u=u, abs_u=u, phi=v)
    adv_v_y = advection_y(dy=dy, v=v, abs_v=v, phi=v)
    adv_v = adv_v_x[0, 0, 0] + adv_v_y[0, 0, 0]

    return adv_u, adv_v


@gtscript.function
def diffusion_x(dx, phi):
    diff_phi = (
        -phi[-2, 0, 0]
        + 16.0 * phi[-1, 0, 0]
        - 30.0 * phi[0, 0, 0]
        + 16.0 * phi[1, 0, 0]
        - phi[2, 0, 0]
    ) / (12.0 * dx**2)
    return diff_phi


@gtscript.function
def diffusion_y(dy, phi):
    diff_phi = (
        -phi[0, -2, 0]
        + 16.0 * phi[0, -1, 0]
        - 30.0 * phi[0, 0, 0]
        + 16.0 * phi[0, 1, 0]
        - phi[0, 2, 0]
    ) / (12.0 * dy**2)
    return diff_phi


@gtscript.function
def diffusion(dx, dy, u, v):
    diff_u_x = diffusion_x(dx=dx, phi=u)
    diff_u_y = diffusion_y(dy=dy, phi=u)
    diff_u = diff_u_x[0, 0, 0] + diff_u_y[0, 0, 0]

    diff_v_x = diffusion_x(dx=dx, phi=v)
    diff_v_y = diffusion_y(dy=dy, phi=v)
    diff_v = diff_v_x[0, 0, 0] + diff_v_y[0, 0, 0]

    return diff_u, diff_v


dtype = np.float64
origin = (3, 3, 0)
rebuild = False
externals = {
    "advection_x": advection_x,
    "advection_y": advection_y,
    "advection": advection,
    "diffusion_x": diffusion_x,
    "diffusion_y": diffusion_y,
    "diffusion": diffusion,
}


def make_rk_stage(backend: str) -> Any:
    backend_opts = {"verbose": False} if backend.startswith("gt") else {}

    @gtscript.stencil(
        backend=backend, externals=externals, rebuild=rebuild, **backend_opts
    )
    def rk_stage(
        in_u_now: gtscript.Field[dtype],
        in_v_now: gtscript.Field[dtype],
        in_u_tmp: gtscript.Field[dtype],
        in_v_tmp: gtscript.Field[dtype],
        out_u: gtscript.Field[dtype],
        out_v: gtscript.Field[dtype],
        *,
        dt: float,
        dx: float,
        dy: float,
        mu: float,
    ):
        with computation(PARALLEL), interval(...):
            adv_u, adv_v = advection(dx=dx, dy=dy, u=in_u_tmp, v=in_v_tmp)
            diff_u, diff_v = diffusion(dx=dx, dy=dy, u=in_u_tmp, v=in_v_tmp)
            out_u = in_u_now[0, 0, 0] + dt * (  # noqa: F841
                -adv_u[0, 0, 0] + mu * diff_u[0, 0, 0]
            )

            out_v = in_v_now[0, 0, 0] + dt * (  # noqa: F841
                -adv_v[0, 0, 0] + mu * diff_v[0, 0, 0]
            )

    return rk_stage


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
    initial = make_initial(nx, ny)
    rk_stage = make_rk_stage(backend)
    dx = 1.0 / (nx - 1)
    dy = dx
    timestep = dx**2

    u_now = gt4py.storage.zeros(
        (nx, ny, 1), dtype, backend=backend, aligned_index=origin
    )
    v_now = gt4py.storage.zeros(
        (nx, ny, 1), dtype, backend=backend, aligned_index=origin
    )
    u_new = gt4py.storage.zeros(
        (nx, ny, 1), dtype, backend=backend, aligned_index=origin
    )
    v_new = gt4py.storage.zeros(
        (nx, ny, 1), dtype, backend=backend, aligned_index=origin
    )
    u_now[:, :, 0] = initial[0]
    v_now[:, :, 0] = initial[1]
    u_new[...] = u_now
    v_new[...] = v_now

    def execute_once() -> None:
        # One stage: base and stencil inputs are identical, outputs are separate.
        rk_stage(
            in_u_now=u_now,
            in_v_now=v_now,
            in_u_tmp=u_now,
            in_v_tmp=v_now,
            out_u=u_new,
            out_v=v_new,
            dt=timestep,
            dx=dx,
            dy=dy,
            mu=0.1,
            origin=origin,
            domain=(nx - 6, ny - 6, 1),
        )

    # Finish any first-call initialization before warmups and timed samples.
    execute_once()
    runtime_ms = median_runtime_ms(execute_once, warmups=warmups, repeats=repeats)
    result = np.stack((u_new[:, :, 0], v_new[:, :, 0]))
    if output is not None:
        result.tofile(output)
    return WorkloadResult(output=result, runtime_ms=runtime_ms)


if __name__ == "__main__":
    parser = workload_parser()
    parser.set_defaults(output=Path("advection_diffusion.bin"))
    parser.add_argument(
        "--backend", choices=("gt:cpu_ifirst", "numpy"), default="gt:cpu_ifirst"
    )
    args = parser.parse_args()
    result = main(**vars(args))
    print(f"NX={args.nx}\nNY={args.ny}\nRUNTIME_MS={result.runtime_ms:.12f}")

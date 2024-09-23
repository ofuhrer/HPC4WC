# ******************************************************
#     Program: stencil2d
#      Author: Tobias Rahn
#       Email: tobias.rahn@inf.ethz.ch
#        Date: 28.06.2024
# Description: 4th-order diffusion
#        Note: Based on https://github.com/ofuhrer/HPC4WC/blob/main/day1/stencil2d.py
# ******************************************************
import click

import numpy as np
from jax import numpy as jnp, config, jit
from functools import partial

import time
from datetime import datetime
import os
import sys


def laplacian(in_field, lap_field, num_halo, extend=0):
    """Compute the Laplacian using 2nd-order centered differences.

    Parameters
    ----------
    in_field : array-like
        Input field (nz x ny x nx with halo in x- and y-direction).
    lap_field : array-like
        Result (must be same size as ``in_field``).
    num_halo : int
        Number of halo points.
    extend : `int`, optional
        Extend computation into halo-zone by this number of points.
    """
    ib = num_halo - extend
    ie = -num_halo + extend
    jb = num_halo - extend
    je = -num_halo + extend

    lap_field = lap_field.at[:, jb:je, ib:ie].set(
        -4.0 * in_field[:, jb:je, ib:ie]
        + in_field[:, jb:je, ib - 1 : ie - 1]
        + in_field[:, jb:je, ib + 1 : ie + 1 if ie != -1 else None]
        + in_field[:, jb - 1 : je - 1, ib:ie]
        + in_field[:, jb + 1 : je + 1 if je != -1 else None, ib:ie]
    )

    return lap_field


def update_halo(field, num_halo):
    """Update the halo-zone using an up/down and left/right strategy.

    Parameters
    ----------
    field : array-like
        Input/output field (nz x ny x nx with halo in x- and y-direction).
    num_halo : int
        Number of halo points.

    Note
    ----
        Corners are updated in the left/right phase of the halo-update.
    """

    # bottom edge (without corners)
    field = field.at[:, :num_halo, num_halo:-num_halo].set(field[:, -2 * num_halo : -num_halo, num_halo:-num_halo])

    # top edge (without corners)
    field = field.at[:, -num_halo:, num_halo:-num_halo].set(field[:, num_halo : 2 * num_halo, num_halo:-num_halo])

    # left edge (including corners)
    field = field.at[:, :, :num_halo].set(field[:, :, -2 * num_halo : -num_halo])

    # right edge (including corners)
    field = field.at[:, :, -num_halo:].set(field[:, :, num_halo : 2 * num_halo])

    return field


def apply_diffusion(in_field, alpha, num_halo, num_iter=1):
    """Integrate 4th-order diffusion equation by a certain number of iterations.

    Parameters
    ----------
    in_field : array-like
        Input field (nz x ny x nx with halo in x- and y-direction).
    lap_field : array-like
        Result (must be same size as ``in_field``).
    alpha : float
        Diffusion coefficient (dimensionless).
    num_iter : `int`, optional
        Number of iterations to execute.
    """

    tmp_field = jnp.empty_like(in_field)
    out_field = jnp.empty_like(in_field)

    for n in range(num_iter):
        in_field = update_halo_jit(in_field, num_halo)

        tmp_field = laplacian_jit(in_field, tmp_field, num_halo=num_halo, extend=1)
        out_field = laplacian_jit(tmp_field, out_field, num_halo=num_halo, extend=0)

        out_field = out_field.at[:, num_halo:-num_halo, num_halo:-num_halo].set(
            in_field[:, num_halo:-num_halo, num_halo:-num_halo]
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]
        )

        if n < num_iter - 1:
            in_field, out_field = out_field, in_field
        else:
            out_field = update_halo_jit(out_field, num_halo)

    return out_field


def calculations(nx, ny, nz, num_iter, num_halo, precision, result_dir="", return_result=False, return_time=False):
    """Driver for apply_diffusion that sets up fields and does timings"""

    assert 0 < nx <= 1024 * 1024, "You have to specify a reasonable value for nx"
    assert 0 < ny <= 1024 * 1024, "You have to specify a reasonable value for ny"
    assert 0 < nz <= 1024, "You have to specify a reasonable value for nz"
    assert 0 < num_iter <= 1024 * 1024, "You have to specify a reasonable value for num_iter"
    assert 2 <= num_halo <= 256, "Your have to specify a reasonable number of halo points"
    alpha = 1.0 / 32.0

    # create jitted functions
    global update_halo_jit, laplacian_jit

    jitter = partial(jit, backend="cpu", inline=True)

    update_halo_jit = jitter(update_halo, static_argnums=(1,))
    laplacian_jit = jitter(laplacian, static_argnums=(2, 3))

    if precision == "64":
        config.update("jax_enable_x64", True)
    else:
        config.update("jax_enable_x64", False)

    in_field = jnp.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo))
    in_field = in_field.at[
        nz // 4 : 3 * nz // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
    ].set(1.0)

    # warmup caches
    apply_diffusion(in_field, alpha, num_halo)

    # time the actual work
    tic = time.time()
    out_field = apply_diffusion(in_field, alpha, num_halo, num_iter=num_iter)
    toc = time.time()

    print(f"Elapsed time for work = {toc - tic} s")

    if result_dir != "":
        result_path = f"{result_dir}/{datetime.now().strftime('%Y%m%dT%H%M%S')}-nx{nx}_ny{ny}_nz{nz}_iter{num_iter}_halo{num_halo}_p{precision}.npy"
        np.save(result_path, out_field)

    if return_time and return_result:
        return out_field, toc - tic

    if return_time:
        return toc - tic

    if return_result:
        return out_field


@click.command()
@click.option("--nx", type=int, required=True, help="Number of gridpoints in x-direction")
@click.option("--ny", type=int, required=True, help="Number of gridpoints in y-direction")
@click.option("--nz", type=int, required=True, help="Number of gridpoints in z-direction")
@click.option("--num_iter", type=int, required=True, help="Number of iterations")
@click.option("--precision", type=click.Choice(["64", "32"]), default="64", required=True, help="Precision")
@click.option(
    "--num_halo",
    type=int,
    default=2,
    help="Number of halo-pointers in x- and y-direction",
)
@click.option(
    "--result_dir",
    type=str,
    default="../data/jax",
    help="Specify the folder where the results should be saved (relative to the location of the script or absolute).",
)
def main(nx, ny, nz, num_iter, result_dir, num_halo, precision):
    calculations(nx, ny, nz, num_iter, num_halo, precision, result_dir=result_dir, return_result=False)


if __name__ == "__main__":
    os.chdir(sys.path[0])  # Change the directory
    main()

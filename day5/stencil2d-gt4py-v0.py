# ******************************************************
#     Program: stencil2d-gt4py
#      Author: Stefano Ubbiali
#       Email: subbiali@phys.ethz.ch
#        Date: 04.06.2020
# Description: GT4Py implementation of 4th-order diffusion
# ******************************************************
import click
import gt4py as gt
from gt4py import gtscript
import matplotlib.pyplot as plt
import numpy as np
import time


@gtscript.function
def laplacian(in_field):
    lap_field = (
        -4.0 * in_field[0, 0, 0]
        + in_field[-1, 0, 0]
        + in_field[1, 0, 0]
        + in_field[0, -1, 0]
        + in_field[0, 1, 0]
    )
    return lap_field


def diffusion_defs(
    in_field: gtscript.Field[float], out_field: gtscript.Field[float], *, alpha: float
):
    from __externals__ import laplacian
    from __gtscript__ import PARALLEL, computation, interval

    with computation(PARALLEL), interval(...):
        lap1 = laplacian(in_field)
        lap2 = laplacian(lap1)
        out_field = in_field - alpha * lap2


def update_halo(field, num_halo):
    # bottom edge (without corners)
    field[num_halo:-num_halo, :num_halo] = field[
        num_halo:-num_halo, -2 * num_halo : -num_halo
    ]

    # top edge (without corners)
    field[num_halo:-num_halo, -num_halo:] = field[
        num_halo:-num_halo, num_halo : 2 * num_halo
    ]

    # left edge (including corners)
    field[:num_halo, :] = field[-2 * num_halo : -num_halo, :]

    # right edge (including corners)
    field[-num_halo:, :] = field[num_halo : 2 * num_halo]


def apply_diffusion(
    diffusion_stencil, in_field, out_field, alpha, num_halo, num_iter=1
):
    # origin and extent of the computational domain
    origin = (num_halo, num_halo, 0)
    domain = (
        in_field.shape[0] - 2 * num_halo,
        in_field.shape[1] - 2 * num_halo,
        in_field.shape[2],
    )

    for n in range(num_iter):
        # halo update
        update_halo(in_field, num_halo)

        # run the stencil
        diffusion_stencil(
            in_field=in_field,
            out_field=out_field,
            alpha=alpha,
            origin=origin,
            domain=domain,
        )

        if n < num_iter - 1:
            # swap input and output fields
            in_field, out_field = out_field, in_field
        else:
            # halo update
            update_halo(out_field, num_halo)


@click.command()
@click.option(
    "--nx", type=int, required=True, help="Number of gridpoints in x-direction"
)
@click.option(
    "--ny", type=int, required=True, help="Number of gridpoints in y-direction"
)
@click.option(
    "--nz", type=int, required=True, help="Number of gridpoints in z-direction"
)
@click.option("--num_iter", type=int, required=True, help="Number of iterations")
@click.option(
    "--num_halo",
    type=int,
    default=2,
    help="Number of halo-points in x- and y-direction",
)
@click.option(
    "--backend", type=str, required=False, default="numpy", help="GT4Py backend."
)
@click.option(
    "--plot_result", type=bool, default=False, help="Make a plot of the result?"
)
def main(nx, ny, nz, num_iter, num_halo=2, backend="numpy", plot_result=False):
    """Driver for apply_diffusion that sets up fields and does timings."""

    assert 0 < nx <= 1024 * 1024, "You have to specify a reasonable value for nx"
    assert 0 < ny <= 1024 * 1024, "You have to specify a reasonable value for ny"
    assert 0 < nz <= 1024, "You have to specify a reasonable value for nz"
    assert (
        0 < num_iter <= 1024 * 1024
    ), "You have to specify a reasonable value for num_iter"
    assert (
        2 <= num_halo <= 256
    ), "You have to specify a reasonable number of halo points"
    assert backend in (
        "numpy",
        "gt:cpu_ifirst",
        "gt:cpu_kfirst",
        "gt:gpu",
        "cuda",
    ), "You have to specify a reasonable value for backend"
    alpha = 1.0 / 32.0

    # default origin
    dorigin = (num_halo, num_halo, 0)

    # allocate input and output fields
    in_field = gt.storage.zeros(
        backend, dorigin, (nx + 2 * num_halo, ny + 2 * num_halo, nz), dtype=float
    )
    out_field = gt.storage.zeros(
        backend, dorigin, (nx + 2 * num_halo, ny + 2 * num_halo, nz), dtype=float
    )

    # prepare input field
    in_field[
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        nz // 4 : 3 * nz // 4,
    ] = 1.0

    # write input field to file
    # swap first and last axes for compatibility with day1/stencil2d.py
    np.save("in_field", np.swapaxes(in_field, 0, 2))

    if plot_result:
        # plot initial field
        plt.ioff()
        plt.imshow(in_field[:, :, 0], origin="lower")
        plt.colorbar()
        plt.savefig("in_field.png")
        plt.close()

    # compile diffusion stencil
    kwargs = {"verbose": True} if backend in ("gtx86", "gtmc", "gtcuda") else {}
    diffusion_stencil = gtscript.stencil(
        definition=diffusion_defs,
        backend=backend,
        externals={"laplacian": laplacian},
        rebuild=False,
        **kwargs,
    )

    # warmup caches
    apply_diffusion(diffusion_stencil, in_field, out_field, alpha, num_halo)

    # time the actual work
    tic = time.time()
    apply_diffusion(
        diffusion_stencil,
        in_field,
        out_field,
        alpha,
        num_halo,
        num_iter=num_iter,
    )
    toc = time.time()
    print(f"Elapsed time for work = {toc - tic} s")

    # save output field
    # swap first and last axes for compatibility with day1/stencil2d.py
    np.save("out_field", np.swapaxes(out_field, 0, 2))

    if plot_result:
        # plot the output field
        plt.ioff()
        plt.imshow(out_field[:, :, 0], origin="lower")
        plt.colorbar()
        plt.savefig("out_field.png")
        plt.close()


if __name__ == "__main__":
    main()

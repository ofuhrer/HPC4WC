# ******************************************************
#     Program: stencil2d-gt4py
#      Author: HPC4WC
#        Date: 11.06.2025
# Description: GT4Py next implementation of 4th-order diffusion
# ******************************************************
from typing import Callable
import time

import click
import gt4py.next as gtx
import matplotlib.pyplot as plt
import numpy as np

backend_str_to_backend = {"None": None, "cpu": gtx.gtfn_cpu, "gpu": gtx.gtfn_gpu}

I = gtx.Dimension("I")
J = gtx.Dimension("J")
K = gtx.Dimension("K")

IJKField = gtx.Field[gtx.Dims[I, J, K], gtx.float64]

# TODO - insert Laplacian

# TODO - implement a single timestep

# TODO - implement ijk-ordered halo-update
# Make sure to use field.ndarray here

# TODO - define apply_diffusion() function

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
    "--backend", type=str, required=False, default="None", help="GT4Py backend."
)
@click.option(
    "--plot_result", type=bool, default=False, help="Make a plot of the result?"
)
def main(nx, ny, nz, num_iter, num_halo=2, backend="None", plot_result=False):
    """Driver for apply_diffusion that sets up fields and does timings."""

    assert 0 < nx <= 1024 * 1024, (
        "You have to specify a reasonable value for nx (0 < nx <= 1024*1024)"
    )
    assert 0 < ny <= 1024 * 1024, (
        "You have to specify a reasonable value for ny (0 < ny <= 1024*1024)"
    )
    assert 0 < nz <= 1024, (
        "You have to specify a reasonable value for nz (0 < nz <= 1024)"
    )
    assert 0 < num_iter <= 1024 * 1024, (
        "You have to specify a reasonable value for num_iter (0 < num_iter <= 1024*1024)"
    )
    assert 2 <= num_halo <= 256, (
        "You have to specify a reasonable number of halo points (2 < num_halo <= 256)"
    )
    assert backend in (
        "None",
        "cpu",
        "gpu",
    ), "You have to specify a reasonable value for backend"

    actual_backend = backend_str_to_backend[backend]

    alpha = 1.0 / 32.0

    # define domain
    # TODO

    # allocate input and output fields
    # TODO

    # prepare input field
    # TODO

    # write input field to file
    # swap first and last axes for compatibility with day1/stencil2d.py
    np.save("in_field", np.swapaxes(in_field.asnumpy(), 0, 2))

    if plot_result:
        # plot initial field
        plt.ioff()
        plt.imshow(in_field.asnumpy()[:, :, in_field.shape[2] // 2], origin="lower")
        plt.colorbar()
        plt.savefig("in_field.png")
        plt.close()

    # TODO - use the selected backend
    diffusion_stencil = # TODO

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
    np.save("out_field", np.swapaxes(out_field.asnumpy(), 0, 2))

    if plot_result:
        # plot the output field
        plt.ioff()
        plt.imshow(out_field.asnumpy()[:, :, out_field.shape[2] // 2], origin="lower")
        plt.colorbar()
        plt.savefig("out_field.png")
        plt.close()


if __name__ == "__main__":
    main()

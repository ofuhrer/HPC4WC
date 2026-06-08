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
OFFSET_PROVIDER = {"_IOff": I, "_JOff": J}


# hpc4wc:student-begin
# hpc4wc:student | # TODO - insert Laplacian
# hpc4wc:student |
# hpc4wc:student | # TODO - implement a single timestep
# hpc4wc:student |
# hpc4wc:student | # TODO - implement ijk-ordered halo-update
# hpc4wc:student | # Make sure to use field.ndarray here
# hpc4wc:student |
# hpc4wc:student | # TODO - define apply_diffusion() function
# hpc4wc:student |
# hpc4wc:student-end
# hpc4wc:solution-begin
@gtx.field_operator
def laplacian(in_field: IJKField) -> IJKField:
    lap_field = (
        -4.0 * in_field
        + in_field(I - 1)
        + in_field(I + 1)
        + in_field(J - 1)
        + in_field(J + 1)
    )
    return lap_field


@gtx.field_operator
def diffusion(in_field: IJKField, alpha: gtx.float64) -> IJKField:
    lap1 = laplacian(in_field)
    lap2 = laplacian(lap1)
    return in_field - alpha * lap2


@gtx.program
def diffusion_program(
    in_field: IJKField,
    out_field: IJKField,
    alpha: gtx.float64,
    nx: gtx.int32,
    ny: gtx.int32,
    nz: gtx.int32,
):
    diffusion(in_field, alpha, out=out_field, domain={I: (0, nx), J: (0, ny), K: (0, nz)})


def update_halo(field: IJKField, num_halo: int):

    # Make sure to use field.ndarray here
    
    # bottom edge (without corners)
    field.ndarray[num_halo:-num_halo, :num_halo] = field.ndarray[
        num_halo:-num_halo, -2 * num_halo : -num_halo
    ]

    # top edge (without corners)
    field.ndarray[num_halo:-num_halo, -num_halo:] = field.ndarray[
        num_halo:-num_halo, num_halo : 2 * num_halo
    ]

    # left edge (including corners)
    field.ndarray[:num_halo, :] = field.ndarray[-2 * num_halo : -num_halo, :]

    # right edge (including corners)
    field.ndarray[-num_halo:, :] = field.ndarray[num_halo : 2 * num_halo]


def apply_diffusion(
    diffusion_stencil: Callable,
    in_field: IJKField,
    out_field: IJKField,
    alpha: gtx.float64,
    num_halo: int,
    num_iter: int = 1,
):
    nx = in_field.shape[0] - 2 * num_halo
    ny = in_field.shape[1] - 2 * num_halo
    nz = in_field.shape[2]

    for n in range(num_iter):
        # halo update
        update_halo(in_field, num_halo)

        # run the stencil
        diffusion_stencil(
            in_field,
            out_field,
            alpha,
            nx,
            ny,
            nz,
            offset_provider=OFFSET_PROVIDER,
        )

        if n < num_iter - 1:
            # swap input and output fields
            in_field, out_field = out_field, in_field
        else:
            # halo update
            update_halo(out_field, num_halo)
# hpc4wc:solution-end


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
    # hpc4wc:student-begin
    # hpc4wc:student | field_domain = None  # TODO
    # hpc4wc:student-end
    # hpc4wc:solution-begin
    field_domain = {
        I: (-num_halo, nx + num_halo),
        J: (-num_halo, ny + num_halo),
        K: (0, nz),
    }
    # hpc4wc:solution-end

    # allocate input and output fields
    # hpc4wc:student-begin
    # hpc4wc:student | in_field = None  # TODO
    # hpc4wc:student | out_field = None  # TODO
    # hpc4wc:student-end
    # hpc4wc:solution-begin
    in_field = gtx.zeros(field_domain, dtype=gtx.float64, allocator=actual_backend)
    out_field = gtx.zeros(field_domain, dtype=gtx.float64, allocator=actual_backend)
    # hpc4wc:solution-end

    # prepare input field
    # hpc4wc:student-begin
    # hpc4wc:student | in_field = None  # TODO
    # hpc4wc:student-end
    # hpc4wc:solution-begin
    in_field[
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        nz // 4 : 3 * nz // 4,
    ] = 1.0
    # hpc4wc:solution-end

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

    # hpc4wc:student-begin
    # hpc4wc:student | # TODO - use the selected backend
    # hpc4wc:student | diffusion_stencil = None  # TODO
    # hpc4wc:student-end
    # hpc4wc:solution-begin
    # select backend
    diffusion_stencil = diffusion_program.with_backend(actual_backend)
    # hpc4wc:solution-end

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

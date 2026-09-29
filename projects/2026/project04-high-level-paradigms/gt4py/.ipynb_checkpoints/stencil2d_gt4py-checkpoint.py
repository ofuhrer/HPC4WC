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
import cupy as cp

backend_str_to_backend = {"None": None, "cpu": gtx.gtfn_cpu, "gpu": gtx.gtfn_gpu}

I = gtx.Dimension("I")
J = gtx.Dimension("J")
K = gtx.Dimension("K")

IJKField = gtx.Field[gtx.Dims[I, J, K], gtx.float64]
#OFFSET_PROVIDER = {"_IOff": I, "_JOff": J}
OFFSET_PROVIDER = {}



@gtx.field_operator
def laplacian(in_field: IJKField) -> IJKField:
    lap_field = (
        -4.0 * in_field + in_field(I - 1) + in_field(I + 1) + in_field(J - 1) + in_field(J + 1)
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

@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True
    )
)
@click.option("--in_field_path", type=str, required=True, help="location of numpy input grid file")
@click.option("--out_field_path", type=str, required=True, help="location of numpy output grid file")
@click.option("--num_iter", type=int, required=True, help="Number of iterations")
@click.option(
    "--num_halo",
    type=int,
    default=2,
    help="Number of halo points in x- and y-direction",
)
@click.option("--device", type=click.Choice(["cpu", "gpu"]), default="cpu", help="set cpu for CPU or gpu for GPU")

def main(in_field_path, out_field_path, num_iter, num_halo=2, device="cpu"):
    """Driver for apply_diffusion that sets up fields and does timings"""

    assert 0 < num_iter <= 1024 * 1024, "You have to specify a reasonable value for num_iter"
    assert 2 <= num_halo <= 256, "You have to specify a reasonable number of halo points"
    assert device in (
        "None",
        "cpu",
        "gpu",
    ), "You have to specify a reasonable value for backend"

    actual_backend = backend_str_to_backend[device]

    alpha = 1.0 / 64.0

    np_in_field = np.load(in_field_path)
    nz, ny_with_halo, nx_with_halo = np_in_field.shape
    ny = ny_with_halo - 2*num_halo
    nx = nx_with_halo - 2*num_halo

    # define domain
    field_domain = {
        I: (-num_halo, nx + num_halo),
        J: (-num_halo, ny + num_halo),
        K: (0, nz),
    }

    # Check data format and type
    np_in_field = np.ascontiguousarray(np_in_field)
    if np_in_field.dtype != np.float64: # Or whatever your stencil uses
        np_in_field = np_in_field.astype(np.float64)


    # allocate input and output fields
    in_field = gtx.zeros(field_domain, dtype=gtx.float64, allocator=actual_backend)
    out_field = gtx.zeros(field_domain, dtype=gtx.float64, allocator=actual_backend)

          
    #copy np data into gt4py object:
    #in_field.ndarray[:] = np_in_field.T
    if device == "gpu":
        in_field.ndarray[:] = cp.asarray(np_in_field.T)
    else:
        in_field.ndarray[:] = np_in_field.T

    # select backend
    diffusion_stencil = diffusion_program.with_backend(actual_backend)
    
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
    print(toc - tic)

    # save output field
    # swap first and last axes for compatibility with day1/stencil2d.py
    np.save(out_field_path, np.swapaxes(out_field.asnumpy(), 0, 2))


if __name__ == "__main__":
    main()

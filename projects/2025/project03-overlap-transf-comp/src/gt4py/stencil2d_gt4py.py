# ******************************************************
#     Program: stencil2d-gt4py
#      Author: HPC4WC
# Description: GT4Py next implementation of 4th-order diffusion
#
#     Changes: Added 'time_stencil' (Philipp Stark)
# ******************************************************
from typing import Callable
import time
import timeit
import os

import struct
import click
import gt4py.next as gtx
import matplotlib.pyplot as plt
import numpy as np

I = gtx.Dimension("I")
J = gtx.Dimension("J")
K = gtx.Dimension("K")

IJKField = gtx.Field[gtx.Dims[I, J, K], gtx.float64]


@gtx.field_operator
def diffusion(
    in_field: IJKField,
    a1: float,
    a2: float,
    a8: float,
    a20: float,
) -> IJKField:
    return (
        a1 * in_field(J - 2)
        + a2 * in_field(I - 1, J - 1)
        + a8 * in_field(J - 1)
        + a2 * in_field(I + 1, J - 1)
        + a1 * in_field(I - 2)
        + a8 * in_field(I - 1)
        + a20 * in_field
        + a8 * in_field(I + 1)
        + a1 * in_field(I + 2)
        + a2 * in_field(I - 1, J + 1)
        + a8 * in_field(J + 1)
        + a2 * in_field(I + 1, J + 1)
        + a1 * in_field(J + 2)
    )

# choose backend based on environment varible if provided
env_var_backend = os.environ.get("USE_BACKEND")
if env_var_backend == "GPU":
    backend = gtx.gtfn_gpu
elif env_var_backend == "CPU":
    backend = gtx.gtfn_cpu
elif env_var_backend is None:  # default case
    backend = gtx.gtfn_cpu
    # backend = gtx.gtfn_gpu
else:
    print(f"Invalid value '{env_var_backend}' in environment variable 'USE_BACKEND'")
    sys.exit(1)

diffusion_stencil = diffusion.with_backend(backend)


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
    interior = gtx.domain(
        {
            I: (0, in_field.shape[0] - 2 * num_halo),
            J: (0, in_field.shape[1] - 2 * num_halo),
            K: (0, in_field.shape[2]),
        }
    )

    for n in range(num_iter):
        # halo update
        update_halo(in_field, num_halo)

        # run the stencil
        diffusion_stencil(
            in_field=in_field,
            out=out_field,
            a1=-alpha,
            a2=-2 * alpha,
            a8=8 * alpha,
            a20=1 - 20 * alpha,
            domain=interior,
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
    "--backend", type=str, required=False, default="None", help="GT4Py backend."
)
@click.option(
    "--plot_result", type=bool, default=False, help="Make a plot of the result?"
)
def main(nx, ny, nz, num_iter, num_halo=2, plot_result=False):
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

    alpha = 1.0 / 32.0

    # define domain
    field_domain = {
        I: (-num_halo, nx + num_halo),
        J: (-num_halo, ny + num_halo),
        K: (0, nz),
    }

    # allocate input and output fields
    in_field = gtx.zeros(field_domain, dtype=gtx.float64, allocator=backend)
    out_field = gtx.zeros(field_domain, dtype=gtx.float64, allocator=backend)

    # prepare input field
    in_field[
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        nz // 4 : 3 * nz // 4,
    ] = 1.0

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


def time_stencil(nx, ny, nz, num_iter, num_halo=2, number=1, repeats=10, verbose=True, incl_transfer=True):
    """Driver for apply_diffusion that sets up fields and does timings."""

    assert 0 < nx <= 1024 * 1024, (
        "You have to specify a reasonable value for nx (0 < nx <= 1024*1024)"
    )
    assert 0 < ny <= 1024 * 1024, (
        "You have to specify a reasonable value for ny (0 < ny <= 1024*1024)"
    )
    assert 0 < nz <= 65536, (
        "You have to specify a reasonable value for nz (0 < nz <= 65536)"
    )
    assert 0 < num_iter <= 1024 * 1024, (
        "You have to specify a reasonable value for num_iter (0 < num_iter <= 1024*1024)"
    )
    assert 2 <= num_halo <= 256, (
        "You have to specify a reasonable number of halo points (2 < num_halo <= 256)"
    )

    alpha = 1.0 / 32.0

    # define domain
    field_domain = {
        I: (-num_halo, nx + num_halo),
        J: (-num_halo, ny + num_halo),
        K: (0, nz),
    }

    # allocate input and output fields
    in_field = gtx.zeros(field_domain, dtype=gtx.float64, allocator=backend)
    out_field = gtx.zeros(field_domain, dtype=gtx.float64, allocator=backend)

    # prepare input field
    in_field[
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        nz // 4 : 3 * nz // 4,
    ] = 1.0

    # select backend
    # diffusion_stencil = diffusion.with_backend(backend)

    # prepare numpy in_field
    in_field_np = in_field.asnumpy()

    def benchmark():
        in_field = gtx.as_field(data=in_field_np, domain=field_domain, allocator=backend)
        apply_diffusion(
            diffusion_stencil,
            in_field,
            out_field,
            alpha,
            num_halo,
            num_iter=num_iter,
        )
        _ = out_field.asnumpy()

    def benchmark_notransfer():
        apply_diffusion(diffusion_stencil, 
                        in_field, 
                        out_field, 
                        alpha, 
                        num_halo, 
                        num_iter=num_iter,
                       )

    # time the actual work
    times = timeit.repeat(benchmark if incl_transfer else benchmark_notransfer, globals=globals(), repeat=repeats, number=number)
        
    avg_time = np.mean(times)
    if verbose:
        # Field dimensions (nx,ny,nz), iterations, time, (unknown for gt4py => -1)
        print(f"{nx}, {ny}, {nz}, {num_iter}, {avg_time}, -1")
    return (nx, ny, nz, num_iter, avg_time)


def write_field_file(filename, field, halosize=0, three=3, sixtyfour=64):
    """
    Write a binary field file compatible with read_field_file from comparison.py
    field: numpy array of shape (zsize, ysize, xsize)
    """
    xsize, ysize, zsize = field.shape
    
    # Swap axes to (zsize, ysize, xsize)
    field_swapped = np.swapaxes(field, 0, 2)
    
    header = struct.pack('6i', three, sixtyfour, halosize, xsize, ysize, zsize)
    data = field_swapped.flatten(order='C')
    data_bytes = struct.pack(f'{data.size}d', *data)
    with open(filename, 'wb') as f:
        f.write(header)
        f.write(data_bytes)

def store_for_comparison(filename, nx=128, ny=128, nz=64, num_iter=512, num_halo=3):
    """
    save result to compare with CUDA version
    """
    alpha = 1.0 / 32.0

    # define domain
    field_domain = {
        I: (-num_halo, nx + num_halo),
        J: (-num_halo, ny + num_halo),
        K: (0, nz),
    }

    # allocate input and output fields
    in_field = gtx.zeros(field_domain, dtype=gtx.float64, allocator=backend)
    out_field = gtx.zeros(field_domain, dtype=gtx.float64, allocator=backend)

    # prepare input field (initial conditions as used in C++/CUDA code)
    in_field[
        1 + nx // 4 : 2 * num_halo - 1 + 3 * nx // 4,
        1 + ny // 4 : 2 * num_halo - 1 + 3 * ny // 4,
        nz // 4 : 3 * nz // 4,
    ] = 1.0

    if 'out' in filename:
        in_np = in_field.asnumpy()
        write_field_file(filename.replace('out', 'in'), in_np, halosize=num_halo)

    apply_diffusion(
        diffusion_stencil,
        in_field,
        out_field,
        alpha,
        num_halo,
        num_iter=num_iter,
    )

    out_np = out_field.asnumpy()
    write_field_file(filename, out_np, halosize=num_halo)

if __name__ == "__main__":
    main()

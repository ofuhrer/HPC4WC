import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import cupy as cp

NUM_THREADS = 1024

# custom laplacian kernel
# 2D grid, 2D block
# shared memory, all elements read from shared memory

_laplacian = cp.RawKernel(
    r"""
extern "C" __global__
void _laplacian(const double* in_field, double* lap_field, int nz, int ny, int nx, int padding) {
    extern __shared__ float shm[];

    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockIdx.z;

    int nx_domain = nx - 2 * padding;
    int ny_domain = ny - 2 * padding;
    // assumes 2.5D distribution
    for(int idz = tidz; idz < nz; idz += gridDim.z) {


        int bidy = blockDim.y * blockIdx.y + padding;
        int bidx = blockDim.x * blockIdx.x + padding;

        int idy = bidy + threadIdx.y;
        int idx = bidx + threadIdx.x;


        // load data into shared memory
        // avoid out of bounds
        if(tidx < nx_domain+1 && tidy < ny_domain+1) {

            shm[
                (threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 1
            ] = in_field[idz*nx*ny + idy*nx + idx];

        }


        // load the for boundaries
        // check out of bounds inside the block
        // top
        if(threadIdx.y == 0) {
            if(bidy + threadIdx.x < ny) {
                shm[
                    (threadIdx.x + 1)*(blockDim.x+2)
                ] = in_field[idz*nx*ny + (bidy + threadIdx.x)*nx + bidx - 1];
            }
        }
        // bottom
        if(threadIdx.y == 1) {
            if(bidy + threadIdx.x < ny) {
                shm[
                    (blockDim.x+2)*(threadIdx.x + 1) + blockDim.x + 1
                ] = in_field[idz*nx*ny + (bidy + threadIdx.x)*nx + bidx + blockDim.x];
            }
        }

        // left
        if(threadIdx.y == 2) {
            if(bidx + threadIdx.x < nx) {
                shm[
                    threadIdx.x + 1
                ] = in_field[idz*nx*ny + (bidy-1)*nx + bidx + threadIdx.x];
            }
        }
        // right
        if(threadIdx.y == 3) {
            if(bidx + threadIdx.x < nx) {
                shm[
                    (blockDim.x + 2) * (blockDim.y+1) + threadIdx.x + 1
                ] = in_field[idz*nx*ny + (bidy + blockDim.y)*nx + bidx + threadIdx.x];
            }
        }


        //sync after load
        __syncthreads();

        if(tidx < nx_domain && tidy < ny_domain) {
            lap_field[idz*nx*ny + idy*nx + idx] = -4.0 * shm[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 1]
                + shm[(threadIdx.y + 1)*(blockDim.x+2) + threadIdx.x + 1 - 1]
                + shm[(threadIdx.y + 1)*(blockDim.x+2) + threadIdx.x + 1 + 1]
                + shm[(threadIdx.y + 1 - 1)*(blockDim.x+2) + threadIdx.x + 1]
                + shm[(threadIdx.y + 1 + 1)*(blockDim.x+2) + threadIdx.x + 1];
        }

        // sync after use
        // only needed if there are not enough blocks to cover the domain
        __syncthreads();
    }
}
""",
    "_laplacian",
)


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
    padding = num_halo - extend
    nz, ny, nx = in_field.shape
    nx_domain = nx - 2 * padding
    ny_domain = ny - 2 * padding

    threads_x = 32
    threads_y = 32
    assert threads_x * threads_y <= NUM_THREADS
    blocks_x = (nx_domain + threads_x - 1) // threads_x
    blocks_y = (ny_domain + threads_y - 1) // threads_y

    # every layer has a 3D tiling
    threads = (threads_x, threads_y, 1)
    # repeat in y to tile z
    blocks = (blocks_x, blocks_y, nz)

    # with 1 layer of padding
    shared_memory_size = (32 + 2) * (32 + 2) * cp.float64().itemsize

    _laplacian(
        blocks,
        threads,
        (in_field, lap_field, nz, ny, nx, padding),
        shared_mem=shared_memory_size,
    )


def halo_update(field, num_halo):
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
    field[:, :num_halo, num_halo:-num_halo] = field[
        :, -2 * num_halo : -num_halo, num_halo:-num_halo
    ]

    # top edge (without corners)
    field[:, -num_halo:, num_halo:-num_halo] = field[
        :, num_halo : 2 * num_halo, num_halo:-num_halo
    ]

    # left edge (including corners)
    field[:, :, :num_halo] = field[:, :, -2 * num_halo : -num_halo]

    # right edge (including corners)
    field[:, :, -num_halo:] = field[:, :, num_halo : 2 * num_halo]


def apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=1):
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
    tmp_field = cp.empty_like(in_field)

    for n in range(num_iter):
        halo_update(in_field, num_halo)

        laplacian(in_field, tmp_field, num_halo=num_halo, extend=1)
        laplacian(tmp_field, out_field, num_halo=num_halo, extend=0)

        out_field[:, num_halo:-num_halo, num_halo:-num_halo] = (
            in_field[:, num_halo:-num_halo, num_halo:-num_halo]
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]
        )

        if n < num_iter - 1:
            in_field, out_field = out_field, in_field
        else:
            halo_update(out_field, num_halo)


def plot_field(field, filename):
    """Plot a 2D field and save it to a file."""
    plt.ioff()
    plt.imshow(field[field.shape[0] // 2, :, :], origin="lower")
    plt.colorbar()
    plt.savefig(filename)
    plt.close()


def main(
    nx,
    ny,
    nz,
    num_iter,
    num_halo=2,
    plot_result=False,
    save_result=True,
    return_result=False,
    benchmark=False,
):
    """Driver for apply_diffusion that sets up fields and does timings"""

    assert 0 < nx <= 1024 * 1024, "You have to specify a reasonable value for nx"
    assert 0 < ny <= 1024 * 1024, "You have to specify a reasonable value for ny"
    assert 0 < nz <= 1024, "You have to specify a reasonable value for nz"
    assert (
        -1 < num_iter <= 1024 * 1024
    ), "You have to specify a reasonable value for num_iter"
    assert (
        2 <= num_halo <= 256
    ), "Your have to specify a reasonable number of halo points"
    alpha = 1.0 / 32.0

    in_field = cp.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo))
    in_field[
        nz // 4 : 3 * nz // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
    ] = 1.0

    out_field = cp.copy(in_field)

    if save_result:
        np.save("in_field", in_field.get())

    if plot_result:
        plot_field(in_field.get(), "in_field.png")

    # warmup caches
    # print("warmup")
    apply_diffusion(in_field, out_field, alpha, num_halo)

    # time the actual work
    # print("starting actual work")
    tic = time.time()
    cp.cuda.Stream.null.synchronize()
    apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=num_iter)
    cp.cuda.Stream.null.synchronize()
    toc = time.time()

    # print(f"Elapsed time for work = {toc - tic} s")

    if save_result:
        np.save("out_field", out_field.get())

    if plot_result:
        plot_field(out_field.get(), "out_field.png")

    if return_result:
        assert not benchmark
        return out_field.get()

    if benchmark:
        return toc - tic

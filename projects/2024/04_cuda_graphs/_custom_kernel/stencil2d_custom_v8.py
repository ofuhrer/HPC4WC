import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import cupy as cp

NUM_THREADS = 1024

# fused laplacian kernel
# custom halo update kernel
# custom update kernel
# 2D grid, 2D block

_fused_laplacian = cp.RawKernel(
    r"""
extern "C" __global__
void _fused_laplacian(const double* __restrict__ in_field, double* __restrict__ out_field, double alpha, int nz, int ny, int nx, int num_halo) {
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockIdx.z;

    int nx_domain = nx - 2 * num_halo;
    int ny_domain = ny - 2 * num_halo;

    for(int idz = tidz; idz < nz; idz += gridDim.z) {
        if(tidx < nx_domain && tidy < ny_domain) {
            int idy = tidy + num_halo;
            int idx = tidx + num_halo;

            double a1 = -1. * alpha;
            double a2 = -2. * alpha;
            double a8 = 8. * alpha;
            double a20 = 1. - 20. * alpha;

            out_field[idz*ny*nx + idy*nx + idx] =
                a1 * in_field[idz*ny*nx + (idy - 2)*nx + idx] +
                a2 * in_field[idz*ny*nx + (idy - 1)*nx + (idx - 1)] +
                a8 * in_field[idz*ny*nx + (idy - 1)*nx + idx] +
                a2 * in_field[idz*ny*nx + (idy - 1)*nx + (idx + 1)] +
                a1 * in_field[idz*ny*nx + idy*nx + (idx - 2)] +
                a8 * in_field[idz*ny*nx + idy*nx + (idx - 1)] +
                a20 * in_field[idz*ny*nx + idy*nx + idx] +
                a8 * in_field[idz*ny*nx + idy*nx + (idx + 1)] +
                a1 * in_field[idz*ny*nx + idy*nx + (idx + 2)] +
                a2 * in_field[idz*ny*nx + (idy + 1)*nx + (idx - 1)] +
                a8 * in_field[idz*ny*nx + (idy + 1)*nx + idx] +
                a2 * in_field[idz*ny*nx + (idy + 1)*nx + (idx + 1)] +
                a1 * in_field[idz*ny*nx + (idy + 2)*nx + idx];

        }


    }
}
""",
    "_fused_laplacian",
)


_halo_update1 = cp.RawKernel(
    r"""
extern "C" __global__
void _halo_update1(double* __restrict__ field, int nz, int ny, int nx, int num_halo) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    for(int i = tid; i < num_halo*(ny - 2 * num_halo)*nz; i += blockDim.x * gridDim.x) {

        int tmp = (num_halo*(ny - 2 * num_halo));
        int idz = i / tmp;
        int j = i % tmp;
        int idnc = j / (ny - 2 * num_halo);
        int idc = j % (ny - 2 * num_halo);

        field[idz*nx*ny + (idc + num_halo)*nx + idnc] = field[
            idz*nx*ny + (idc + num_halo)*nx + idnc + nx - 2*num_halo
        ];
        field[idz*nx*ny + (idc + num_halo)*nx + idnc + nx - num_halo] = field[
            idz*nx*ny + (idc + num_halo)*nx + idnc + num_halo
        ];


    }
}
""",
    "_halo_update1",
)

_halo_update2 = cp.RawKernel(
    r"""
extern "C" __global__
void _halo_update2(double* __restrict__ field, int nz, int ny, int nx, int num_halo) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    for(int i = tid; i < num_halo*(ny)*nz; i += blockDim.x * gridDim.x) {

        int tmp = (num_halo*(ny));
        int idz = i / tmp;
        int j = i % tmp;
        int idnc = j / (ny);
        int idc = j % (ny);

        field[idz*nx*ny + idnc*nx + idc] = field[
            idz*nx*ny + (idnc+ny-2*num_halo)*nx + idc
        ];
        field[idz*nx*ny + (idnc+ny-num_halo)*nx + idc] = field[
            idz*nx*ny + (idnc+num_halo)*nx + idc
        ];

    }
}
""",
    "_halo_update2",
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
    nx = field.shape[2]
    ny = field.shape[1]
    nz = field.shape[0]

    assert nx == ny, "Only square domains are supported"

    blocks_1 = ((nz * num_halo * (nx - 2 * num_halo) + NUM_THREADS - 1) // NUM_THREADS,)

    _halo_update1(blocks_1, (NUM_THREADS,), (field, nz, ny, nx, num_halo))

    blocks_2 = ((nz * num_halo * (nx) + NUM_THREADS - 1) // NUM_THREADS,)

    _halo_update2(blocks_2, (NUM_THREADS,), (field, nz, ny, nx, num_halo))


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

    nz, ny, nx = in_field.shape
    nx_domain = nx - 2 * num_halo
    ny_domain = ny - 2 * num_halo

    threads_x = 32
    threads_y = 32
    assert threads_x * threads_y <= NUM_THREADS
    blocks_x = (nx_domain + threads_x - 1) // threads_x
    blocks_y = (ny_domain + threads_y - 1) // threads_y

    # every layer has a 2D tiling
    threads = (threads_x, threads_y, 1)
    # repeat in y to tile z
    blocks = (blocks_x, blocks_y, nz)

    for n in range(num_iter):
        halo_update(in_field, num_halo)

        _fused_laplacian(
            blocks, threads, (in_field, out_field, alpha, nz, ny, nx, num_halo)
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

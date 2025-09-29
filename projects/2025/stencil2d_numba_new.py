import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time

from numba import njit  # serial nopython JIT

DTYPE = np.float32

@njit(parallel=False, fastmath=False, cache=False)
def laplacian(in_field, lap_field, num_halo, extend=0):
    nz, ny, nx = in_field.shape
    ib = num_halo - extend
    ie = nx - num_halo + extend
    jb = num_halo - extend
    je = ny - num_halo + extend
    for k in range(nz):
        for j in range(jb, je):
            for i in range(ib, ie):
                lap_field[k, j, i] = (
                    -4.0 * in_field[k, j, i]
                    + in_field[k, j, i - 1]
                    + in_field[k, j, i + 1]
                    + in_field[k, j - 1, i]
                    + in_field[k, j + 1, i]
                )

@njit(parallel=False, fastmath=False, cache=False)
def update_halo(field, num_halo):
    nz, ny, nx = field.shape
    h = num_halo
    for k in range(nz):
        # bottom/top (without corners)
        for j in range(h):
            for i in range(h, nx - h):
                field[k, j, i] = field[k, ny - 2*h + j, i]
                field[k, ny - h + j, i] = field[k, h + j, i]
        # left/right (including corners)
        for j in range(ny):
            for i in range(h):
                field[k, j, i] = field[k, j, nx - 2*h + i]
                field[k, j, nx - h + i] = field[k, j, h + i]

@njit(parallel=False, fastmath=False, cache=False)
def axpy_diffusion_step(in_field, lap2_field, out_field, alpha, num_halo):
    h = num_halo
    nz, ny, nx = in_field.shape
    for k in range(nz):
        for j in range(h, ny - h):
            for i in range(h, nx - h):
                out_field[k, j, i] = in_field[k, j, i] - alpha * lap2_field[k, j, i]

@njit(parallel=False, fastmath=False, cache=False)
def apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=1):
    tmp_field = np.empty_like(in_field)
    a = in_field
    b = out_field
    for n in range(num_iter):
        update_halo(a, num_halo)
        laplacian(a, tmp_field, num_halo=num_halo, extend=1)
        laplacian(tmp_field, b, num_halo=num_halo, extend=0)
        axpy_diffusion_step(a, b, b, alpha, num_halo)
        if n < num_iter - 1:
            t = a; a = b; b = t
    update_halo(b, num_halo)
    # copy back into out_field to ensure the caller buffer holds result
    nz, ny, nx = out_field.shape
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                out_field[k, j, i] = b[k, j, i]

@click.command()
@click.option("--nx", type=int, required=True)
@click.option("--ny", type=int, required=True)
@click.option("--nz", type=int, required=True)
@click.option("--num_iter", type=int, required=True)
@click.option("--num_halo", type=int, default=2)
@click.option("--plot_result", is_flag=True, default=False)
def main(nx, ny, nz, num_iter, num_halo=2, plot_result=False):
    assert 0 < nx <= 1024 * 1024
    assert 0 < ny <= 1024 * 1024
    assert 0 < nz <= 1024
    assert 0 < num_iter <= 1024 * 1024
    assert 2 <= num_halo <= 256

    alpha = DTYPE(1.0 / 32.0)

    in_field = np.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo), dtype=DTYPE)
    in_field[
        nz // 4 : 3 * nz // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
    ] = 1.0
    out_field = in_field.copy()

    if plot_result:
        plt.ioff()
        mid = in_field.shape[0] // 2
        plt.imshow(in_field[mid, :, :], origin="lower")
        plt.colorbar(); plt.savefig("in_field_numba.png"); plt.close()

    # warmup JIT (no file I/O)
    _in_w = in_field.copy()
    _out_w = out_field.copy()
    apply_diffusion(_in_w, _out_w, alpha, num_halo, num_iter=1)

    # timed run
    tic = time.time()
    apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=num_iter)
    toc = time.time()
    print(f"Elapsed time for work = {toc - tic:.6f} s")

    if plot_result:
        mid = out_field.shape[0] // 2
        plt.imshow(out_field[mid, :, :], origin="lower")
        plt.colorbar(); plt.savefig("out_field_numba.png"); plt.close()

if __name__ == "__main__":
    main()

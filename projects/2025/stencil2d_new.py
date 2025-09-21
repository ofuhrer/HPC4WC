# ******************************************************
# Program: stencil2d-numpy (4th-order diffusion baseline)
# ******************************************************
import click
import matplotlib
matplotlib.use("Agg") # so it runs headless and renders images in memory and writes them straight to files
import matplotlib.pyplot as plt
import numpy as np
import time


def laplacian(in_field: np.ndarray, lap_field: np.ndarray, num_halo: int, extend: int = 0):
    """
    Compute the 2D 5-point Laplacian (second order).
    Writes into lap_field; only the target slice is written.
    extend=1 extends computation one cell into the halo (for the 2nd Laplacian),
    extend=0 computes only the interior.
    """
    h = num_halo
    ib = h - extend
    ie = -h + extend
    jb = h - extend
    je = -h + extend

    lap_field[:, jb:je, ib:ie] = (
        -4.0 * in_field[:, jb:je, ib:ie]
        + in_field[:, jb:je, ib - 1 : ie - 1]
        + in_field[:, jb:je, ib + 1 : ie + 1 if ie != -1 else None]
        + in_field[:, jb - 1 : je - 1, ib:ie]
        + in_field[:, jb + 1 : je + 1 if je != -1 else None, ib:ie]
    )


def update_halo(field: np.ndarray, num_halo: int):
    """
    Periodic halo update (up/down then left/right; corners in second phase).
    Shape: [nz, ny+2h, nx+2h]
    """
    h = num_halo
    # bottom/top (without corners)
    field[:, :h, h:-h] = field[:, -2 * h : -h, h:-h]
    field[:, -h:, h:-h] = field[:, h : 2 * h, h:-h]
    # left/right (including corners)
    field[:, :, :h] = field[:, :, -2 * h : -h]
    field[:, :, -h:] = field[:, :, h : 2 * h]


def axpy_diffusion_step(in_field: np.ndarray, lap2_field: np.ndarray,
                        out_field: np.ndarray, alpha: float, num_halo: int):
    """
    out = in - alpha * lap2  (interior only)
    """
    h = num_halo
    out_field[:, h:-h, h:-h] = in_field[:, h:-h, h:-h] - alpha * lap2_field[:, h:-h, h:-h]

    # factored update into own function to easy switch integratiors or fuse operations recommended by CHATgpt


def apply_diffusion(in_field: np.ndarray, out_field: np.ndarray,
                    alpha: float, num_halo: int, num_iter: int = 1):
    """
    Run num_iter steps. Ensures 'out_field' holds the final result.
    Uses pointer swapping; copies only if num_iter is even.
    """
    a = in_field
    b = out_field
    tmp = np.empty_like(in_field)

    for n in range(num_iter):
        update_halo(a, num_halo)
        laplacian(a, tmp, num_halo=num_halo, extend=1)
        laplacian(tmp, b, num_halo=num_halo, extend=0)
        axpy_diffusion_step(a, b, b, alpha, num_halo)
        if n < num_iter - 1:
            a, b = b, a

    # final halo for the result currently in b
    update_halo(b, num_halo)

    # Guarantee the caller's out_field buffer contains the result
    if (num_iter % 2) == 0:
        out_field[...] = in_field  # even iter: result resides in 'in_field'


@click.command()
@click.option("--nx", type=int, required=True, help="Gridpoints in x (without halos)")
@click.option("--ny", type=int, required=True, help="Gridpoints in y (without halos)")
@click.option("--nz", type=int, required=True, help="Gridpoints in z")
@click.option("--num_iter", type=int, required=True, help="Number of iterations")
@click.option("--num_halo", type=int, default=2, help="Halo width")
@click.option("--plot_result", is_flag=True, default=False, help="Save PNGs of mid-slice")
def main(nx, ny, nz, num_iter, num_halo=2, plot_result=False):
    """Driver that sets up fields, warms up, times, and writes results."""
    assert 0 < nx <= 1024 * 1024
    assert 0 < ny <= 1024 * 1024
    assert 0 < nz <= 1024
    assert 0 < num_iter <= 1024 * 1024
    assert 2 <= num_halo <= 256

    alpha = np.float32(1.0 / 32.0)  # used float32 instead of default float 64 from stecil2d.py
                                    # to match fortrans wp=4, leads to less memory traffic

    in_field = np.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo), dtype=np.float32)
    in_field[
        nz // 4 : 3 * nz // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
    ] = 1.0
    out_field = in_field.copy()

    np.save("in_field_numpy", in_field)

    if plot_result:
        plt.ioff()
        mid = in_field.shape[0] // 2
        plt.imshow(in_field[mid, :, :], origin="lower")
        plt.colorbar()
        plt.savefig("in_field_numpy.png")
        plt.close()

    # warmup (avoid polluting measured arrays)
    _in_w = in_field.copy()
    _out_w = out_field.copy()
    apply_diffusion(_in_w, _out_w, alpha, num_halo, num_iter=1)

    # timed run
    tic = time.time()
    apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=num_iter)
    toc = time.time()
    print(f"Elapsed time for work = {toc - tic:.6f} s")

    np.save("out_field_numpy", out_field)

    if plot_result:
        mid = out_field.shape[0] // 2
        plt.imshow(out_field[mid, :, :], origin="lower")
        plt.colorbar()
        plt.savefig("out_field_numpy.png")
        plt.close()


if __name__ == "__main__":
    main()

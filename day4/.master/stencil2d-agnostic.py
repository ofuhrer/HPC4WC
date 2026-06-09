# hpc4wc:student-begin
# hpc4wc:student | # ******************************************************
# hpc4wc:student | #     Program: stencil2d-cupy
# hpc4wc:student | #      Author: Stefano Ubbiali, Oliver Fuhrer
# hpc4wc:student | #       Email: subbiali@phys.ethz.ch, ofuhrer@ethz.ch
# hpc4wc:student | #        Date: 04.06.2020
# hpc4wc:student | # Description: CuPy implementation of 4th-order diffusion
# hpc4wc:student | # ******************************************************
# hpc4wc:student | import click
# hpc4wc:student | import matplotlib
# hpc4wc:student |
# hpc4wc:student | matplotlib.use("Agg")
# hpc4wc:student | import matplotlib.pyplot as plt
# hpc4wc:student | import numpy as np
# hpc4wc:student | import time
# hpc4wc:student |
# hpc4wc:student | # TODO: make this file run with CuPy when available and NumPy otherwise.
# hpc4wc:student |
# hpc4wc:student |
# hpc4wc:student |
# hpc4wc:student | def laplacian(in_field, lap_field, num_halo, extend=0):
# hpc4wc:student |     """ Compute the Laplacian using 2nd-order centered differences.
# hpc4wc:student |
# hpc4wc:student |     Parameters
# hpc4wc:student |     ----------
# hpc4wc:student |     in_field : array-like
# hpc4wc:student |         Input field (nz x ny x nx with halo in x- and y-direction).
# hpc4wc:student |     lap_field : array-like
# hpc4wc:student |         Result (must be same size as ``in_field``).
# hpc4wc:student |     num_halo : int
# hpc4wc:student |         Number of halo points.
# hpc4wc:student |     extend : `int`, optional
# hpc4wc:student |         Extend computation into halo-zone by this number of points.
# hpc4wc:student |     """
# hpc4wc:student |     ib = num_halo - extend
# hpc4wc:student |     ie = -num_halo + extend
# hpc4wc:student |     jb = num_halo - extend
# hpc4wc:student |     je = -num_halo + extend
# hpc4wc:student |
# hpc4wc:student |     lap_field[:, jb:je, ib:ie] = (
# hpc4wc:student |         -4.0 * in_field[:, jb:je, ib:ie]
# hpc4wc:student |         + in_field[:, jb:je, ib - 1 : ie - 1]
# hpc4wc:student |         + in_field[:, jb:je, ib + 1 : ie + 1 if ie != -1 else None]
# hpc4wc:student |         + in_field[:, jb - 1 : je - 1, ib:ie]
# hpc4wc:student |         + in_field[:, jb + 1 : je + 1 if je != -1 else None, ib:ie]
# hpc4wc:student |     )
# hpc4wc:student |
# hpc4wc:student |
# hpc4wc:student | def update_halo(field, num_halo):
# hpc4wc:student |     """ Update the halo-zone using an up/down and left/right strategy.
# hpc4wc:student |
# hpc4wc:student |     Parameters
# hpc4wc:student |     ----------
# hpc4wc:student |     field : array-like
# hpc4wc:student |         Input/output field (nz x ny x nx with halo in x- and y-direction).
# hpc4wc:student |     num_halo : int
# hpc4wc:student |         Number of halo points.
# hpc4wc:student |
# hpc4wc:student |     Note
# hpc4wc:student |     ----
# hpc4wc:student |         Corners are updated in the left/right phase of the halo-update.
# hpc4wc:student |     """
# hpc4wc:student |     # bottom edge (without corners)
# hpc4wc:student |     field[:, :num_halo, num_halo:-num_halo] = field[
# hpc4wc:student |         :, -2 * num_halo : -num_halo, num_halo:-num_halo
# hpc4wc:student |     ]
# hpc4wc:student |
# hpc4wc:student |     # top edge (without corners)
# hpc4wc:student |     field[:, -num_halo:, num_halo:-num_halo] = field[
# hpc4wc:student |         :, num_halo : 2 * num_halo, num_halo:-num_halo
# hpc4wc:student |     ]
# hpc4wc:student |
# hpc4wc:student |     # left edge (including corners)
# hpc4wc:student |     field[:, :, :num_halo] = field[:, :, -2 * num_halo : -num_halo]
# hpc4wc:student |
# hpc4wc:student |     # right edge (including corners)
# hpc4wc:student |     field[:, :, -num_halo:] = field[:, :, num_halo : 2 * num_halo]
# hpc4wc:student |
# hpc4wc:student |
# hpc4wc:student | def apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=1):
# hpc4wc:student |     """ Integrate 4th-order diffusion equation by a certain number of iterations.
# hpc4wc:student |
# hpc4wc:student |     Parameters
# hpc4wc:student |     ----------
# hpc4wc:student |     in_field : array-like
# hpc4wc:student |         Input field (nz x ny x nx with halo in x- and y-direction).
# hpc4wc:student |     lap_field : array-like
# hpc4wc:student |         Result (must be same size as ``in_field``).
# hpc4wc:student |     alpha : float
# hpc4wc:student |         Diffusion coefficient (dimensionless).
# hpc4wc:student |     num_iter : `int`, optional
# hpc4wc:student |         Number of iterations to execute.
# hpc4wc:student |     """
# hpc4wc:student |     tmp_field = np.empty_like(in_field)
# hpc4wc:student |
# hpc4wc:student |     for n in range(num_iter):
# hpc4wc:student |         update_halo(in_field, num_halo)
# hpc4wc:student |
# hpc4wc:student |         laplacian(in_field, tmp_field, num_halo=num_halo, extend=1)
# hpc4wc:student |         laplacian(tmp_field, out_field, num_halo=num_halo, extend=0)
# hpc4wc:student |
# hpc4wc:student |         out_field[:, num_halo:-num_halo, num_halo:-num_halo] = (
# hpc4wc:student |             in_field[:, num_halo:-num_halo, num_halo:-num_halo]
# hpc4wc:student |             - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]
# hpc4wc:student |         )
# hpc4wc:student |
# hpc4wc:student |         if n < num_iter - 1:
# hpc4wc:student |             in_field, out_field = out_field, in_field
# hpc4wc:student |         else:
# hpc4wc:student |             update_halo(out_field, num_halo)
# hpc4wc:student |
# hpc4wc:student |
# hpc4wc:student | @click.command()
# hpc4wc:student | @click.option(
# hpc4wc:student |     "--nx", type=int, required=True, help="Number of gridpoints in x-direction"
# hpc4wc:student | )
# hpc4wc:student | @click.option(
# hpc4wc:student |     "--ny", type=int, required=True, help="Number of gridpoints in y-direction"
# hpc4wc:student | )
# hpc4wc:student | @click.option(
# hpc4wc:student |     "--nz", type=int, required=True, help="Number of gridpoints in z-direction"
# hpc4wc:student | )
# hpc4wc:student | @click.option("--num_iter", type=int, required=True, help="Number of iterations")
# hpc4wc:student | @click.option(
# hpc4wc:student |     "--num_halo",
# hpc4wc:student |     type=int,
# hpc4wc:student |     default=2,
# hpc4wc:student |     help="Number of halo points in x- and y-direction",
# hpc4wc:student | )
# hpc4wc:student | @click.option(
# hpc4wc:student |     "--plot_result", type=bool, default=False, help="Make a plot of the result?"
# hpc4wc:student | )
# hpc4wc:student | def main(nx, ny, nz, num_iter, num_halo=2, plot_result=False):
# hpc4wc:student |     """Driver for apply_diffusion that sets up fields and does timings"""
# hpc4wc:student |
# hpc4wc:student |     assert 0 < nx <= 1024 * 1024, "You have to specify a reasonable value for nx"
# hpc4wc:student |     assert 0 < ny <= 1024 * 1024, "You have to specify a reasonable value for ny"
# hpc4wc:student |     assert 0 < nz <= 1024, "You have to specify a reasonable value for nz"
# hpc4wc:student |     assert (
# hpc4wc:student |         0 < num_iter <= 1024 * 1024
# hpc4wc:student |     ), "You have to specify a reasonable value for num_iter"
# hpc4wc:student |     assert (
# hpc4wc:student |         2 <= num_halo <= 256
# hpc4wc:student |     ), "You have to specify a reasonable number of halo points"
# hpc4wc:student |     alpha = 1.0 / 32.0
# hpc4wc:student |
# hpc4wc:student |     in_field = np.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo))
# hpc4wc:student |     in_field[
# hpc4wc:student |         nz // 4 : 3 * nz // 4,
# hpc4wc:student |         num_halo + ny // 4 : num_halo + 3 * ny // 4,
# hpc4wc:student |         num_halo + nx // 4 : num_halo + 3 * nx // 4,
# hpc4wc:student |     ] = 1.0
# hpc4wc:student |
# hpc4wc:student |     out_field = np.copy(in_field)
# hpc4wc:student |
# hpc4wc:student |     np.save("in_field", in_field)
# hpc4wc:student |
# hpc4wc:student |     if plot_result:
# hpc4wc:student |         plt.ioff()
# hpc4wc:student |         plt.imshow(in_field[in_field.shape[0] // 2, :, :], origin="lower")
# hpc4wc:student |         plt.colorbar()
# hpc4wc:student |         plt.savefig("in_field.png")
# hpc4wc:student |         plt.close()
# hpc4wc:student |
# hpc4wc:student |     # warmup caches
# hpc4wc:student |     apply_diffusion(in_field, out_field, alpha, num_halo)
# hpc4wc:student |
# hpc4wc:student |     # time the actual work
# hpc4wc:student |     tic = time.time()
# hpc4wc:student |     apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=num_iter)
# hpc4wc:student |     toc = time.time()
# hpc4wc:student |
# hpc4wc:student |     print(f"Elapsed time for work = {toc - tic} s")
# hpc4wc:student |
# hpc4wc:student |     np.save("out_field", out_field)
# hpc4wc:student |
# hpc4wc:student |     if plot_result:
# hpc4wc:student |         plt.imshow(out_field[out_field.shape[0] // 2, :, :], origin="lower")
# hpc4wc:student |         plt.colorbar()
# hpc4wc:student |         plt.savefig("out_field.png")
# hpc4wc:student |         plt.close()
# hpc4wc:student |
# hpc4wc:student |
# hpc4wc:student | if __name__ == "__main__":
# hpc4wc:student |     main()
# hpc4wc:student-end
# hpc4wc:solution-begin
# ******************************************************
#     Program: stencil2d-cupy
#      Author: Stefano Ubbiali, Oliver Fuhrer
#       Email: subbiali@phys.ethz.ch, ofuhrer@ethz.ch
#        Date: 04.06.2020
# Description: CuPy implementation of 4th-order diffusion
# ******************************************************
import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time

try:
    import cupy as xp
    print("Running on GPU with CuPy")
except ImportError:
    xp = np
    print("Running on CPU with numpy")


def get_asnumpy(z):
    try:
        return z.get()
    except AttributeError:
        return z


def laplacian(in_field, lap_field, num_halo, extend=0):
    """ Compute the Laplacian using 2nd-order centered differences.

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

    lap_field[:, jb:je, ib:ie] = (
        -4.0 * in_field[:, jb:je, ib:ie]
        + in_field[:, jb:je, ib - 1 : ie - 1]
        + in_field[:, jb:je, ib + 1 : ie + 1 if ie != -1 else None]
        + in_field[:, jb - 1 : je - 1, ib:ie]
        + in_field[:, jb + 1 : je + 1 if je != -1 else None, ib:ie]
    )


def halo_update(field, num_halo):
    """ Update the halo-zone using an up/down and left/right strategy.

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
    """ Integrate 4th-order diffusion equation by a certain number of iterations.

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
    tmp_field = xp.empty_like(in_field)

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
    help="Number of halo points in x- and y-direction",
)
@click.option(
    "--plot_result", type=bool, default=False, help="Make a plot of the result?"
)
def main(nx, ny, nz, num_iter, num_halo=2, plot_result=False):
    """Driver for apply_diffusion that sets up fields and does timings"""

    assert 0 < nx <= 1024 * 1024, "You have to specify a reasonable value for nx"
    assert 0 < ny <= 1024 * 1024, "You have to specify a reasonable value for ny"
    assert 0 < nz <= 1024, "You have to specify a reasonable value for nz"
    assert (
        0 < num_iter <= 1024 * 1024
    ), "You have to specify a reasonable value for num_iter"
    assert (
        2 <= num_halo <= 256
    ), "You have to specify a reasonable number of halo points"
    alpha = 1.0 / 32.0

    in_field = xp.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo))
    in_field[
        nz // 4 : 3 * nz // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
    ] = 1.0

    out_field = xp.copy(in_field)

    np.save("in_field", get_asnumpy(in_field))

    if plot_result:
        plt.ioff()
        plt.imshow(get_asnumpy(in_field[in_field.shape[0] // 2, :, :]), origin="lower")
        plt.colorbar()
        plt.savefig("in_field.png")
        plt.close()

    # warmup caches
    apply_diffusion(in_field, out_field, alpha, num_halo)

    # time the actual work
    tic = time.time()
    apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=num_iter)
    toc = time.time()

    print(f"Elapsed time for work = {toc - tic} s")

    np.save("out_field", get_asnumpy(out_field))

    if plot_result:
        plt.ioff()
        plt.imshow(get_asnumpy(out_field[out_field.shape[0] // 2, :, :]), origin="lower")
        plt.colorbar()
        plt.savefig("out_field.png")
        plt.close()


if __name__ == "__main__":
    main()
# hpc4wc:solution-end

"""
This is just a copy of the code from the slides for benchmarking.
"""

import numpy as np


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
            + in_field[:, jb:je, ib - 1: ie - 1]
            + in_field[:, jb:je, ib + 1: ie + 1 if ie != -1 else None]
            + in_field[:, jb - 1: je - 1, ib:ie]
            + in_field[:, jb + 1: je + 1 if je != -1 else None, ib:ie]
    )


def update_halo(field, num_halo):
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
                                              :, -2 * num_halo: -num_halo, num_halo:-num_halo
                                              ]

    # top edge (without corners)
    field[:, -num_halo:, num_halo:-num_halo] = field[
                                               :, num_halo: 2 * num_halo, num_halo:-num_halo
                                               ]

    # left edge (including corners)
    field[:, :, :num_halo] = field[:, :, -2 * num_halo: -num_halo]

    # right edge (including corners)
    field[:, :, -num_halo:] = field[:, :, num_halo: 2 * num_halo]


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
    tmp_field = np.empty_like(in_field)

    for n in range(num_iter):
        update_halo(in_field, num_halo)

        laplacian(in_field, tmp_field, num_halo=num_halo, extend=1)
        laplacian(tmp_field, out_field, num_halo=num_halo, extend=0)

        out_field[:, num_halo:-num_halo, num_halo:-num_halo] = (
                in_field[:, num_halo:-num_halo, num_halo:-num_halo]
                - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]
        )

        if n < num_iter - 1:
            in_field, out_field = out_field, in_field
        else:
            update_halo(out_field, num_halo)


def baseline_stencil(in_field, out_field, num_halo, nx, ny, nz, num_iter, tmp_field, alpha):
    apply_diffusion(in_field, out_field, alpha, num_halo)

    return out_field


def main(nx, ny, nz, num_iter, num_halo=2):
    """Driver for apply_diffusion that sets up fields"""
    alpha = 1.0 / 32.0

    in_field = np.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo))
    in_field[
    nz // 4: 3 * nz // 4,
    num_halo + ny // 4: num_halo + 3 * ny // 4,
    num_halo + nx // 4: num_halo + 3 * nx // 4,
    ] = 1.0

    out_field = np.copy(in_field)

    apply_diffusion(in_field, out_field, alpha, num_halo)


if __name__ == "__main__":
    main()

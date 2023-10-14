# ******************************************************
#     Program: stencil2d-cupy
#      Author: Stefano Ubbiali, Oliver Fuhrer
#       Email: subbiali@phys.ethz.ch, ofuhrer@ethz.ch
#        Date: 04.06.2020
# Description: CuPy implementation of 4th-order diffusion
# ******************************************************
import numpy as np

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

def update_halo(field, num_halo):
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
    field[:, :num_halo, num_halo:-num_halo] = field[:, -2 * num_halo : -num_halo, num_halo:-num_halo]

    # top edge (without corners)
    field[:, -num_halo:, num_halo:-num_halo] = field[:, num_halo : 2 * num_halo, num_halo:-num_halo]

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
    num_iter : `int`, optionals
        Number of iterations to execute.
    """
    (zsize, ysize, xsize) = np.shape(in_field)
    nx = xsize - 2*num_halo
    ny = ysize - 2*num_halo
    nz = zsize
    tmp_field = np.zeros((ysize, xsize))
    # print(nz, ny, nx, xsize, ysize, np.shape(tmp_field))

    for iter in range(num_iter):
        update_halo(in_field, num_halo)

        for k in range(0, nz):
            for j in range(num_halo - 1, ny + num_halo + 1):
                for i in range(num_halo - 1, nx + num_halo + 1):
                    # tmp_field[i, j] = -4.0 * in_field[i, j, k]+ in_field[i - 1, j, k]+ in_field[i + 1, j, k]+ in_field[i, j - 1, k]+ in_field[i, j + 1, k]
                    tmp_field[j, i] = -4.0 * in_field[k, j, i] + in_field[k, j, i - 1] + in_field[k, j, i + 1] + in_field[k, j - 1, i] + in_field[k, j + 1, i]

            for j in range(num_halo, ny + num_halo):
                for i in range(num_halo, nx + num_halo):
                    # laplap = -4.0 * tmp_field[i, j]+ tmp_field[i - 1, j]+ tmp_field[i + 1, j]+ tmp_field[i, j - 1]+ tmp_field[i, j + 1]
                    laplap = -4.0 * tmp_field[j, i] + tmp_field[j, i - 1] + tmp_field[j, i + 1] + tmp_field[j - 1, i] + tmp_field[j + 1, i]
                    if iter != num_iter - 1:
                        in_field[k, j, i] = in_field[k, j, i] - alpha * laplap
                    else:
                        out_field[k, j, i] = in_field[k, j, i] - alpha * laplap



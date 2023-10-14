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

    for k in range(in_field.shape[0]):
        for i in range(jb, je):
            for j in range(ib, ie):
                lap_field[k, i, j] = (
                    -4.0 * in_field[k, i, j]
                    + in_field[k, i, j - 1]
                    + in_field[k, i, j + 1] if j + 1 != ie else 0
                    + in_field[k, i - 1, j]
                    + in_field[k, i + 1, j] if i + 1 != je else 0
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
    for k in range(field.shape[0]):
        # Top-left corner
        for i in range(num_halo):
            for j in range(num_halo, field.shape[2] - num_halo):
                field[k, i, j] = field[k, -2 * num_halo + i, j]

        # Top edge (without corners)
        for i in range(field.shape[1] - num_halo, field.shape[1]):
            for j in range(num_halo, field.shape[2] - num_halo):
                field[k, i, j] = field[k, num_halo + i - field.shape[1], j]

        # Left edge (including corners)
        for i in range(field.shape[1]):
            for j in range(num_halo):
                field[k, i, j] = field[k, i, -2 * num_halo + j]

        # Right edge (including corners)
        for i in range(field.shape[1]):
            for j in range(field.shape[2] - num_halo, field.shape[2]):
                field[k, i, j] = field[k, i, num_halo + j - field.shape[2]]

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

    for k in range(out_field.shape[0]):
        for i in range(num_halo, out_field.shape[1] - num_halo):
            for j in range(num_halo, out_field.shape[2] - num_halo):
                out_field[k, i, j] = (
                    in_field[k, i, j]
                    - alpha * out_field[k, i, j]
                )

        if n < num_iter - 1:
            for k in range(out_field.shape[0]):
                for i in range(out_field.shape[1]):
                    for j in range(out_field.shape[2]):
                        in_field[k, i, j], out_field[k, i, j] = out_field[k, i, j], in_field[k, i, j]
        else:
            update_halo(out_field, num_halo)
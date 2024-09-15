# -*- coding: utf-8 -*-
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
rank_size = comm.Get_size()


def periodic(phi, nx, nb):
    """ Make the 2-d array ``phi`` periodic along the first dimension.

    At the left and right border the number of ``nb`` points is overwritten.
    The periodicity of this operation is ``nx``.
    Based on periodic.m from the full isentropic model in MATLAB, 2014.

    Parameters
    ----------
    phi : np.ndarray
        The input 2-d array.
    nx : int
        Number of horizontal grid points.
    nb : int
        Number of horizontal boundary layers.

    Returns
    -------
    np.ndarray :
        The input array ``phi`` made periodic.

    Examples
    --------
    >>> nx = 100
    >>> nb = 3
    >>> nz = 60
    >>> phi = np.random.rand(nx + 2 * nb, nz)
    >>> phi = periodic(phi, nx, nb)
    """
    phi[0:nb, :] = phi[nx : nx + nb, :]
    phi[-nb:, :] = phi[-nx - nb : -nx, :]

    return phi


def relax(phi, nx, nb, phi1, phi2):
    """ Relaxation of boundary conditions.

    Based on relax.m from the full isentropic model in MATLAB, 2014.

    Parameters
    ----------
    phi : np.ndarray
        The input 2-d array whose boundary will be relaxed.
    nx : int
        The number of horizontal grid points.
    nb : int
        The number of horizontal boundary layers.
    phi1 : float
        Value of ``phi`` along the left boundary.
    phi2 : float
        Value of ``phi`` along the right boundary.

    Returns
    -------
    np.ndarray :
        The input array ``phi`` with relaxed boundary.

    Examples
    --------
    >>> nx = 100
    >>> nb = 3
    >>> nz = 60
    >>> phi = np.random.rand(nx + 2 * nb, nz)
    >>> phi1 = 0.0
    >>> phi2 = 1.0
    >>> phi = relax(phi, nx, nb, phi1, phi2)
    """

    # relaxation is done over nr grid points
    nr = 8
    n = 2 * nb + nx

    # initialize relaxation array
    rel = np.array([1, 0.99, 0.95, 0.8, 0.5, 0.2, 0.05, 0.01])

    # relaxation boundary conditions
    if rank == 0:
        if len(phi.shape) == 2:
            for i in range(0, nr):
                phi[i, :] = phi1 * rel[i] + phi[i, :] * (1 - rel[i])
        else:
            for i in range(0, nr):
                phi[i] = phi1 * rel[i] + phi[i] * (1 - rel[i])
    elif rank == rank_size - 1 :
        if len(phi.shape) == 2:
            for i in range(0, nr):
                phi[n - 1 - i, :] = phi2 * rel[i] + phi[n - 1 - i, :] * (1 - rel[i])
        else:
            for i in range(0, nr):
                phi[n - 1 - i] = phi2 * rel[i] + phi[n - 1 - i] * (1 - rel[i])

    return phi


# END OF BOUNDARY.PY

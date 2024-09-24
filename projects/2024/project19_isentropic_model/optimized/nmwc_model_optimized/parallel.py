from nmwc_model_optimized.namelist import nb, irelax
import numpy as np
from mpi4py import MPI
from typing import Union

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
rank_size = comm.Get_size()


def exchange_borders_2d(data_p: np.ndarray, tag: int):
    """ Exchange borders with neighboring processes for 2-dimensional dataset `data` that was scattered along the first axis.

    Parameters
    ----------
    data : np.ndarray
        Two-dimensional dataset of a process whose borders must be exchanged with the neighboring ranks.
    tag : int
        Unique identifier of the variable, used as a tag to send and receive data between processes.
    """
    left_rank = (rank - 1) % rank_size
    right_rank = (rank + 1) % rank_size

    send_to_right = data_p[-2*nb:-nb, :]
    send_to_left = data_p[nb:2*nb, :]

    new_left_border = np.empty((nb, data_p.shape[1]), dtype=data_p.dtype)
    new_right_border = np.empty((nb, data_p.shape[1]), dtype=data_p.dtype)

    # Send to left, receive from right
    comm.Sendrecv(sendbuf=send_to_left, dest=left_rank, sendtag=rank * 10_000 + 100 * left_rank +
                  tag, recvbuf=new_right_border, source=right_rank, recvtag=right_rank * 10_000 + 100 * rank + tag)

    # Send to right, receive from left
    comm.Sendrecv(sendbuf=send_to_right, dest=right_rank, sendtag=rank * 10_000 + 100 * right_rank +
                  tag, recvbuf=new_left_border, source=left_rank, recvtag=left_rank * 10_000 + 100 * rank + tag)

    # do not copy left border if not periodic and rank is left-outermost
    if irelax == 0 or rank != 0:
        data_p[:nb, :] = new_left_border[:, :]

    # do not copy right border if not periodic and rank is right-outermost
    if irelax == 0 or rank != rank_size - 1:
        data_p[-nb:, :] = new_right_border[:, :]

    return data_p


def gather_1d(data_p: np.ndarray, data_g: Union[np.ndarray, None], nx_p: int, s: slice):
    """ Gather a one-dimensional dataset onto the process with rank 0.

    Parameters
    ----------
    data_p : np.ndarray
        Local data of a process, to be gathered into `data_g` on the process with rank 0.
    data_g : np.ndarray
        Variable into which the data is to be gathered on the process with rank 0. Can be `None`
        on all ranks, except rank 0.
    nx_p : int
        The size along the first dimension of the gathered data for one process.
    s : slice
        Slice to be used to slice `data_p` and insert the data into `data_g`, e.g. to get rid of borders.
        WARNING: this function uses the same slice for both the scattered and the gathered data!
    """
    # Note: `nx_p` could be calculated from slice + data, but doing that would be slower
    if rank == 0:
        buffer = np.empty((rank_size, nx_p), dtype=data_g.dtype)
    else:
        buffer = None

    comm.Gather(data_p[s], buffer, root=0)

    if rank == 0:
        data_g[s] = buffer.flatten()

        # exchange leftmost & rightmost border points if periodic
        if irelax == 0:
            data_g[:nb] = data_g[-2*nb:-nb]
            data_g[-nb:] = data_g[nb:2*nb]

    return data_g


def gather_2d(data_p: np.ndarray, data_g: Union[np.ndarray, None], nx_p: int, s: slice):
    """ Gather a two-dimensional dataset onto the process with rank 0.

    Parameters
    ----------
    data_p : np.ndarray
        Local data of a process, to be gathered into `data_g` on the process with rank 0.
    data_g : np.ndarray
        Variable into which the data is to be gathered on the process with rank 0. Can be `None`
        on all ranks, except rank 0.
    nx_p : int
        The size along the first dimension of the gathered data for one process.
    s : slice
        Slice to be used to slice `data_p` and insert the data into `data_g`, e.g. to get rid of borders.
        WARNING: this function uses the same slice for both the scattered and the gathered data!
    """

    # Note: `nx_p` could be calculated from slice + data, but doing that would be slower
    if rank == 0:
        buffer = np.empty(
            (rank_size * nx_p, data_g.shape[1]), dtype=data_g.dtype)
    else:
        buffer = None

    comm.Gather(data_p[s, :], buffer, root=0)

    if rank == 0:
        data_g[s, :] = buffer[:, :]

        # exchange leftmost & rightmost border points if periodic
        if irelax == 0:
            pass
            data_g[:nb, :] = buffer[-nb:, :]
            data_g[-nb:, :] = buffer[:nb, :]

    return data_g

import numpy as np
from mpi4py import MPI

from nmwc_model_optimized.parallel import exchange_borders_2d

def test_exchange_borders():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    gathered_data = np.arange(128, dtype=int).reshape((16, -1))
    nb = 2
    nx = 3

    # construct slice where we don't know the borders
    rank_data = gathered_data[rank*(nx):(rank+1)*nx + 2 * nb, :]
    rank_data[:nb, :] = 0
    rank_data[-nb:, :] = 0

    # exchange borders
    exchange_borders_2d(rank_data, 12)

    # each rank should now know its borders
    np.testing.assert_array_equal(rank_data, gathered_data[rank*(nx):(rank+1)*nx + 2 * nb, :])

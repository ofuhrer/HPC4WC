import numpy as np
from mpi4py import MPI

from nmwc_model_optimized.parallel import gather_2d
from nmwc_model_optimized.namelist import nb

def test_gather_2d():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank_size = comm.Get_size()
    nx_p = 8

    # Create a periodic array
    gathered_data = np.arange(128 * (rank_size * nx_p + 2 * nb), dtype=int) \
        .reshape((rank_size * nx_p + 2 * nb, -1))
    gathered_data[:nb, :] = gathered_data[-2*nb:-nb, :]
    gathered_data[-nb:, :] = gathered_data[nb:2*nb, :]

    # Slice taken from `solver.py`
    start_index = rank * nx_p
    end_index = (rank + 1) * nx_p + 2 * nb
    rank_slice = slice(start_index, end_index)
    
    # Construct process-specific slice
    rank_data = gathered_data[rank_slice]

    if rank == 0:
        result = np.empty_like(gathered_data)
    else:
        result = None

    # We want to gather the data onto rank 0 and each process should not send it's border values
    gather_2d(rank_data, result, nx_p, slice(nb, -nb))
    
    if rank == 0:
        np.testing.assert_array_equal(gathered_data, result)

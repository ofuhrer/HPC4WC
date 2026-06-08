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
import cupy as cp
from mpi4py import MPI

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
    extend : int, optional
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

def update_halo(field, num_halo, comm):
    """ Update halo regions of the field using MPI communication.

    Parameters
    ----------
    field : array-like
        Field to update (nz x ny x nx with halo in x- and y-direction).
    num_halo : int
        Number of halo points.
    comm : MPI.Comm
        MPI communicator.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size > 1:
        if rank > 0:
            send_buf = cp.asnumpy(field[:, num_halo:2*num_halo, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=rank-1, sendtag=0,
                          recvbuf=recv_buf, source=rank-1, recvtag=1)
            field[:, :num_halo, :] = cp.asarray(recv_buf)
        if rank < size - 1:
            send_buf = cp.asnumpy(field[:, -2*num_halo:-num_halo, :])
            recv_buf = np.empty_like(send_buf)
            comm.Sendrecv(send_buf, dest=rank+1, sendtag=1,
                          recvbuf=recv_buf, source=rank+1, recvtag=0)
            field[:, -num_halo:, :] = cp.asarray(recv_buf)

    field[:, :, :num_halo] = field[:, :, -2 * num_halo : -num_halo]
    field[:, :, -num_halo:] = field[:, :, num_halo : 2 * num_halo]

def apply_diffusion(in_field, out_field, alpha, num_halo, num_iter, comm):
    """ Apply diffusion to the input field for a given number of iterations.

    Parameters
    ----------
    in_field : array-like
        Input field (nz x ny x nx with halo in x- and y-direction).
    out_field : array-like
        Output field (must be same size as ``in_field``).
    alpha : float
        Diffusion coefficient.
    num_halo : int
        Number of halo points.
    num_iter : int
        Number of iterations to perform.
    comm : MPI.Comm
        MPI communicator.
    """
    tmp_field = cp.empty_like(in_field)

    for n in range(num_iter):
        update_halo(in_field, num_halo, comm)

        laplacian(in_field, tmp_field, num_halo=num_halo, extend=1)
        laplacian(tmp_field, out_field, num_halo=num_halo, extend=0)

        out_field[:, num_halo:-num_halo, num_halo:-num_halo] = (
            in_field[:, num_halo:-num_halo, num_halo:-num_halo]
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]
        )

        if n < num_iter - 1:
            in_field, out_field = out_field, in_field
        else:
            update_halo(out_field, num_halo, comm)

@click.command()
@click.option("--nx", type=int, required=True, help="Number of gridpoints in x-direction")
@click.option("--ny", type=int, required=True, help="Number of gridpoints in y-direction")
@click.option("--nz", type=int, required=True, help="Number of gridpoints in z-direction")
@click.option("--num_iter", type=int, required=True, help="Number of iterations")
@click.option("--num_halo", type=int, default=2, help="Number of halo-pointers in x- and y-direction")
@click.option("--plot_result", type=bool, default=False, help="Make a plot of the result?")
def main(nx, ny, nz, num_iter, num_halo=2, plot_result=False):
    """ Main function to run the diffusion simulation.

    Parameters
    ----------
    nx : int
        Number of gridpoints in x-direction.
    ny : int
        Number of gridpoints in y-direction.
    nz : int
        Number of gridpoints in z-direction.
    num_iter : int
        Number of iterations to perform.
    num_halo : int, optional
        Number of halo-pointers in x- and y-direction.
    plot_result : bool, optional
        Whether to plot the result.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    assert 0 < nx <= 1024 * 1024, "You have to specify a reasonable value for nx"
    assert 0 < ny <= 1024 * 1024, "You have to specify a reasonable value for ny"
    assert 0 < nz <= 1024, "You have to specify a reasonable value for nz"
    assert 0 < num_iter <= 1024 * 1024, "You have to specify a reasonable value for num_iter"
    assert 2 <= num_halo <= 256, "Your have to specify a reasonable number of halo points"
    alpha = 1.0 / 32.0

    local_nz = nz // size
    start_z = rank * local_nz
    end_z = start_z + local_nz

    cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()

    in_field = cp.zeros((local_nz, ny + 2 * num_halo, nx + 2 * num_halo), dtype=cp.float64)
    if nz // 4 <= start_z < 3 * nz // 4 or nz // 4 < end_z <= 3 * nz // 4 or (start_z < nz // 4 and end_z > 3 * nz // 4):
        z_start = max(0, nz // 4 - start_z)
        z_end = min(local_nz, 3 * nz // 4 - start_z)
        in_field[z_start:z_end,
                 num_halo + ny // 4 : num_halo + 3 * ny // 4,
                 num_halo + nx // 4 : num_halo + 3 * nx // 4] = 1.0

    out_field = cp.copy(in_field)

    apply_diffusion(in_field, out_field, alpha, num_halo, 1, comm)

    comm.Barrier()
    tic = time.time()
    apply_diffusion(in_field, out_field, alpha, num_halo, num_iter, comm)
    comm.Barrier()
    toc = time.time()

    if rank == 0:
        print(f"Elapsed time for work = {toc - tic} s")

    local_out_field = cp.asnumpy(out_field[:, num_halo:-num_halo, num_halo:-num_halo])
    full_out_field = None
    if rank == 0:
        full_out_field = np.empty((nz, ny, nx))

    comm.Gather(local_out_field, full_out_field, root=0)

    if rank == 0:
        # Pad the full_out_field with zeros
        padded_out_field = np.pad(full_out_field, 
                                  ((0, 0), (num_halo, num_halo), (num_halo, num_halo)),
                                  mode='constant', 
                                  constant_values=0)

        np.save("out_field", padded_out_field)
        if plot_result:
            plt.imshow(full_out_field[full_out_field.shape[0] // 2, :, :], origin="lower")
            plt.colorbar()
            plt.savefig("out_field.png")
            plt.close()

if __name__ == "__main__":
    main()
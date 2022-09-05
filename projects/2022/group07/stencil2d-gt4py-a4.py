# ******************************************************
#     Program: stencil2d-gt4py-a4.py
#     Authors: Ruben Strässle, Yilu Chen, Marc Federer
#     (based on stencil2d-gt4py-v3.py by Stefano Ubbiali)
#     Description: Second version of MPI standard halo updates 
#       using a stencil object to copy halo points. 
# ******************************************************

import warnings
warnings.simplefilter("ignore", category=UserWarning)

import click
import gt4py as gt
from gt4py import gtscript
import matplotlib.pyplot as plt
import numpy as np
import time
from copy import deepcopy
import sys

from mpi4py import MPI
from partitioner import Partitioner

@gtscript.function
def laplacian(in_field):
    lap_field = (
        -4.0 * in_field[0, 0, 0]
        + in_field[-1, 0, 0]
        + in_field[1, 0, 0]
        + in_field[0, -1, 0]
        + in_field[0, 1, 0]
    )
    return lap_field


def diffusion_defs(
    in_field: gtscript.Field[float], out_field: gtscript.Field[float], *, alpha: float
):
    from __externals__ import laplacian
    from __gtscript__ import PARALLEL, computation, interval

    with computation(PARALLEL), interval(...):
        lap1 = laplacian(in_field)
        lap2 = laplacian(lap1)
        out_field = in_field - alpha * lap2
        
def copy_defs(src: gtscript.Field["dtype"], dst: gtscript.Field["dtype"]):
    from __gtscript__ import PARALLEL, computation, interval

    with computation(PARALLEL), interval(...):
        dst = src
    
def update_halo_mpi(p, copy_stencil, field, num_halo, backend):
    
    # allocate recv buffers and pre-post the receives (top and bottom edge, without corners)
    b_rcvbuf = np.ascontiguousarray(np.empty_like(np.asarray(field[num_halo:-num_halo, 0:num_halo, :])))
    t_rcvbuf = np.ascontiguousarray(np.empty_like(np.asarray(field[num_halo:-num_halo, -num_halo:, :])))
    reqs_tb = []
    reqs_tb.append(p.comm().Irecv(b_rcvbuf, source = p.bottom()))
    reqs_tb.append(p.comm().Irecv(t_rcvbuf, source = p.top()))

    # allocate recv buffers and pre-post the receives (left and right edge, including corners)
    l_rcvbuf = np.ascontiguousarray(np.empty_like(np.asarray(field[0:num_halo, :, :])))
    r_rcvbuf = np.ascontiguousarray(np.empty_like(np.asarray(field[-num_halo:, :, :])))
    reqs_lr = []
    reqs_lr.append(p.comm().Irecv(l_rcvbuf, source = p.left()))
    reqs_lr.append(p.comm().Irecv(r_rcvbuf, source = p.right()))
    
    # pack and send (top and bottom edge, without corners)
    b_sndbuf = np.asarray(field[num_halo:-num_halo, -2 * num_halo:-num_halo, :]).copy()
    reqs_tb.append(p.comm().Isend(b_sndbuf, dest = p.top()))
    t_sndbuf = np.asarray(field[num_halo:-num_halo, num_halo:2 * num_halo, :]).copy()
    reqs_tb.append(p.comm().Isend(t_sndbuf, dest = p.bottom()))
    
    # wait and unpack
    for req in reqs_tb:
        req.wait()
    
    # field dims without halo points
    nx_p = field.shape[0]-2*num_halo
    ny_p = field.shape[1]-2*num_halo
    nz_p = field.shape[2]
    
    # copy bottom halo points
    b_rcvbuf_storage = gt.storage.from_array(b_rcvbuf, backend = backend, default_origin = (0,0,0))
    copy_stencil(
        src=b_rcvbuf_storage,
        dst=field,
        origin={"src": (0, 0, 0), "dst": (num_halo, 0, 0)},
        domain=(nx_p, num_halo, nz_p),
    )

    # copy top halo points
    t_rcvbuf_storage = gt.storage.from_array(t_rcvbuf, backend = backend, default_origin = (0,0,0))
    copy_stencil(
        src=t_rcvbuf_storage,
        dst=field,
        origin={"src": (0, 0, 0), "dst": (num_halo, ny_p+num_halo, 0)},
        domain=(nx_p, num_halo, nz_p),
    )
    
    # pack and send (left and right edge, including corners)
    l_sndbuf = np.asarray(field[-2 * num_halo:-num_halo, :, :]).copy()
    reqs_lr.append(p.comm().Isend(l_sndbuf, dest = p.right()))
    r_sndbuf = np.asarray(field[num_halo:2 * num_halo, :, :]).copy()
    reqs_lr.append(p.comm().Isend(r_sndbuf, dest = p.left()))

    # wait and unpack
    for req in reqs_lr:
        req.wait()

    # copy left halo points
    l_rcvbuf_storage = gt.storage.from_array(l_rcvbuf, backend = backend, default_origin = (0,0,0))
    copy_stencil(
        src=l_rcvbuf_storage,
        dst=field,
        origin={"src": (0, 0, 0), "dst": (0, 0, 0)},
        domain=(num_halo, ny_p + 2*num_halo, nz_p),
    )
    
    # copy right halo points
    r_rcvbuf_storage = gt.storage.from_array(r_rcvbuf, backend = backend, default_origin = (0,0,0))
    copy_stencil(
        src=r_rcvbuf_storage,
        dst=field,
        origin={"src": (0, 0, 0), "dst": (nx_p+num_halo, 0, 0)},
        domain=(num_halo, ny_p+2*num_halo, nz_p),
    )
    
def apply_diffusion(
    p, backend, diffusion_stencil, copy_stencil, in_field, out_field, alpha, num_halo, num_iter=1
):
    # origin and extent of the computational domain
    origin = (num_halo, num_halo, 0)
    domain = (
        in_field.shape[0] - 2 * num_halo,
        in_field.shape[1] - 2 * num_halo,
        in_field.shape[2],
    )

    for n in range(num_iter):
        # halo update
        update_halo_mpi(p, copy_stencil, in_field, num_halo, backend)

        # run the stencil
        diffusion_stencil(
            in_field=in_field,
            out_field=out_field,
            alpha=alpha,
            origin=origin,
            domain=domain,
        )

        if n < num_iter - 1:
            # swap input and output fields
            in_field, out_field = out_field, in_field
        else:
            # halo update
            update_halo_mpi(p, copy_stencil, in_field, num_halo, backend)


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
    help="Number of halo-points in x- and y-direction",
)
@click.option(
    "--backend", type=str, required=False, default="numpy", help="GT4Py backend."
)
@click.option(
    "--plot_result", type=bool, default=False, help="Make a plot of the result?"
)
def main(nx, ny, nz, num_iter, num_halo=2, backend="numpy", plot_result=False):
    """Driver for apply_diffusion that sets up fields and does timings."""
    
    assert 0 < nx <= 1024 * 1024, "You have to specify a reasonable value for nx"
    assert 0 < ny <= 1024 * 1024, "You have to specify a reasonable value for ny"
    assert 0 < nz <= 1024, "You have to specify a reasonable value for nz"
    assert (
        0 < num_iter <= 1024 * 1024
    ), "You have to specify a reasonable value for num_iter"
    assert (
        2 <= num_halo <= 256
    ), "You have to specify a reasonable number of halo points"
    assert backend in (
        "numpy",
        "gt:cpu_ifirst",
        "gt:cpu_kfirst",
        "gt:gpu",
        "cuda",
    ), "You have to specify a reasonable value for backend"
    alpha = 1.0 / 32.0

    # default origin
    dorigin = (num_halo, num_halo, 0)
    
    # initialize partitioner
    comm = MPI.COMM_WORLD
    p = Partitioner(comm, [nz, ny, nx], num_halo)
    
    size = p.num_ranks()
    assert nx % size == 0, "Cannot partition given grid size with given number of ranks (nx)"
    assert ny % size == 0, "Cannot partition given grid size with given number of ranks (ny)"

    # prepare global field in numpy
    if p.rank() == 0:
        f = np.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo), dtype = np.float64)
        f[nz // 4:3 * nz // 4, num_halo + ny // 4:num_halo + 3 * ny // 4, num_halo + nx // 4:num_halo + 3 * nx // 4] = 1.0
        np.save("in_field", f)
    else:
        f = None
    
    # split np fields to ranks
    in_field = p.scatter(f)
    # change axis order to (nx,ny,nz) for gt4py
    in_field = np.transpose(in_field, (2,1,0))
    out_field = np.copy(in_field)
    
    # allocate input and output fields in storage containers for each rank
    in_field = gt.storage.from_array(in_field, backend, dorigin)
    out_field = gt.storage.from_array(out_field, backend, dorigin)
    
    if plot_result and p.rank() == 0:
        # plot initial field
        plt.ioff()
        plt.imshow(f[f.shape[0] // 2, :, :], origin="lower")
        plt.colorbar()
        plt.savefig("in_field.png")
        plt.close()
            
    # compile diffusion stencil
    kwargs = {"verbose": True} if backend in ("gtx86", "gtmc", "gtcuda") else {}
   
    diffusion_stencil = gtscript.stencil(
        definition=diffusion_defs,
        backend=backend,
        externals={"laplacian": laplacian},
        rebuild=False,
        **kwargs,
    )

    # compile copy stencil
    copy_stencil = gtscript.stencil(
        definition=copy_defs,
        backend=backend,
        dtypes={"dtype": np.float64},
        rebuild=False,
        **kwargs,
    )
    
    # warmup caches
    apply_diffusion(p, backend, diffusion_stencil, copy_stencil, in_field, out_field, alpha, num_halo)
    
    comm.Barrier()
    
    # time the actual work
    tic = time.time()
    apply_diffusion(
        p,
        backend,
        diffusion_stencil,
        copy_stencil,
        in_field,
        out_field,
        alpha,
        num_halo,
        num_iter=num_iter,
    )
    toc = time.time()
    comm.Barrier()
    if p.rank() == 0:
        print(f"Elapsed time for work = {toc - tic} s")
    
    # change axis order to (nz,ny,nx) for partitioner class
    out_field = np.transpose(out_field, (2,1,0))
    f = p.gather(out_field)

    if p.rank() == 0:
        np.save("out_field_a4", f) #changed name
        if plot_result:
            plt.imshow(np.asarray(f)[f.shape[0] // 2, :, :], origin="lower")
            plt.colorbar()
            plt.savefig("out_field_a4.png") #changed name
            plt.close()
    
if __name__ == "__main__":
    main()


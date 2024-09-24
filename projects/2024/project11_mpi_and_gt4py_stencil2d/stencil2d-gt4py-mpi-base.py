# ******************************************************
#     Program: stencil2d-gt4py
#      Author: Stefano Ubbiali
#       Email: subbiali@phys.ethz.ch
#        Date: 04.06.2020
# Description: GT4Py implementation of 4th-order diffusion
# ******************************************************
import setuptools
import click
import gt4py as gt
from gt4py import gtscript
import matplotlib.pyplot as plt
import numpy as np
import time
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
    in_field: gtscript.Field["dtype"],
    out_field: gtscript.Field["dtype"],
    *,
    alpha: float,
):
    from __externals__ import laplacian
    from __gtscript__ import PARALLEL, computation, interval

    with computation(PARALLEL), interval(...):
        lap1 = laplacian(in_field)
        lap2 = laplacian(lap1)
        out_field = in_field - alpha * lap2
        
def update_halo(field, num_halo,p):
    
    # allocate recv buffers and pre-post the receives (top and bottom edge, without corners)
    b_rcvbuf = np.ascontiguousarray(np.empty_like(np.asarray(field[num_halo:-num_halo, 0:num_halo, :])))
    t_rcvbuf = np.ascontiguousarray(np.empty_like(np.asarray(field[num_halo:-num_halo, -num_halo:, :])))
    reqs_tb = []
    reqs_tb.append(p.comm().Irecv(b_rcvbuf, source = p.bottom()))
    reqs_tb.append(p.comm().Irecv(t_rcvbuf, source = p.top()))

    # allocate recv buffers and pre-post the receives (left and right edge, including corners)
    l_rcvbuf = np.ascontiguousarray(np.empty_like(np.asarray(field[0:num_halo, :, :])))
    r_rcvbuf = np.ascontiguousarray( np.empty_like(np.asarray(field[-num_halo:, :, :])))
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

    # copy halo points bottom and top
    field[num_halo:-num_halo, 0:num_halo, :] = b_rcvbuf
    field[num_halo:-num_halo, -num_halo:, :] = t_rcvbuf
    
    # pack and send (left and right edge, including corners)
    l_sndbuf = np.asarray(field[-2 * num_halo:-num_halo, :, :]).copy()
    reqs_lr.append(p.comm().Isend(l_sndbuf, dest = p.right()))
    r_sndbuf = np.asarray(field[num_halo:2 * num_halo, :, :]).copy()
    reqs_lr.append(p.comm().Isend(r_sndbuf, dest = p.left()))

    # wait and unpack
    for req in reqs_lr:
        req.wait()

    # copy halo points left and right
    field[0:num_halo, :, :] = l_rcvbuf
    field[-num_halo:, :, :] = r_rcvbuf

def apply_diffusion(
    diffusion_stencil, in_field, out_field, alpha,
    num_halo, num_iter=1, p=None):
    # origin and extent of the computational domain
    origin = (num_halo, num_halo, 0)
    domain = (
        in_field.shape[0] - 2 * num_halo,
        in_field.shape[1] - 2 * num_halo,
        in_field.shape[2],
    )

    for n in range(num_iter):
        # halo update
        update_halo(in_field, num_halo,p)

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
            update_halo(out_field, num_halo,p)


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
    
    comm = MPI.COMM_WORLD

    p = Partitioner(comm, [nz, ny, nx], num_halo)
    
    if p.rank() == 0:
        field= np.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo))
        field[nz // 4:3 * nz // 4, num_halo + ny // 4:num_halo + 3 * ny // 4, num_halo + nx // 4:num_halo + 3 * nx // 4] = 1.0
    else:
        field = None

    # allocate input and output fields
    np_field = p.scatter(field)
    
    # default origin
    dorigin = (num_halo, num_halo, 0)
    
    # allocate input and output fields in gt4py
    in_field = gt.storage.from_array(np.swapaxes(np_field, 0, 2), backend, dorigin)
    out_field = gt.storage.empty(backend, dorigin, shape=in_field.shape, dtype=in_field.dtype)
    # write input field to file
    # swap first and last axes for compliance with F-layout
    field_in = p.gather(np.swapaxes(in_field, 0, 2))
    if p.rank() == 0:
      np.save("in_field-gt4py-mpi-base", field_in)

      if plot_result:
          # plot initial field
          plt.ioff()
          plt.imshow(np.asarray(field_in[field_in.shape[0] // 2, :, :]), origin="lower")
          plt.colorbar()
          plt.savefig("in_field-gt4py-mpi-base.png")
          plt.close()

    # compile diffusion stencil
    kwargs = {"verbose": True} if backend in ("gtx86", "gtmc", "gtcuda") else {}
    diffusion_stencil = gtscript.stencil(
        definition=diffusion_defs,
        backend=backend,
        dtypes={"dtype": float},
        externals={"laplacian": laplacian},
        rebuild=False,
        **kwargs,
    )

    # warmup caches
    apply_diffusion(
        diffusion_stencil,in_field, out_field, alpha, num_halo,p=p)

    
    # time the actual work
    comm.Barrier()
    tic = time.time()
    apply_diffusion(
        diffusion_stencil,
        in_field,
        out_field,
        alpha,
        num_halo,
        num_iter=num_iter,
        p=p
    )
    toc = time.time()
    comm.Barrier()
    if p.rank() == 0:
      print(f"Elapsed time for work = {toc - tic} s")

    # save output field
    # swap first and last axes for compliance with F-layout
    field_out = p.gather(np.swapaxes(out_field, 0, 2))
    if p.rank() == 0:
      np.save("out_field-gt4py-mpi-base", field_out)

      if plot_result:
          # plot the output field
          plt.ioff()
          plt.imshow(np.asarray(field_out[field_out.shape[0] // 2, :, :]), origin="lower")
          plt.colorbar()
          plt.savefig("out_field-gt4py-mpi-base.png")
          plt.close()
                   
if __name__ == "__main__":
    main()

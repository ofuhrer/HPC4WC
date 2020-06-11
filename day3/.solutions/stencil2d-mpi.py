# ******************************************************
#     Program: stencil2d
#      Author: Oliver Fuhrer
#       Email: oliverf@vulcan.com
#        Date: 20.05.2020
# Description: Simple stencil example
# ******************************************************

import time
import math
import numpy as np
import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
from partitioner import Partitioner


def laplacian( in_field, lap_field, num_halo, extend=0 ):
    """Compute Laplacian using 2nd-order centered differences.
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    num_halo  -- number of halo points
    
    Keyword arguments:
    extend    -- extend computation into halo-zone by this number of points
    """

    ib = num_halo - extend
    ie = - num_halo + extend
    jb = num_halo - extend
    je = - num_halo + extend
    
    lap_field[:, jb:je, ib:ie] = - 4. * in_field[:, jb:je, ib:ie]  \
        + in_field[:, jb:je, ib - 1:ie - 1] + in_field[:, jb:je, ib + 1:ie + 1 if ie != -1 else None]  \
        + in_field[:, jb - 1:je - 1, ib:ie] + in_field[:, jb + 1:je + 1 if je != -1 else None, ib:ie]


def update_halo( field, num_halo, p=None ):
    """Update the halo-zone using an up/down and left/right strategy.
    
    field    -- input/output field (nz x ny x nx with halo in x- and y-direction)
    num_halo -- number of halo points
    
    Note: corners are updated in the left/right phase of the halo-update
    """

    # allocate recv buffers and pre-post the receives (top and bottom edge, without corners)
    b_rcvbuf = np.empty_like(field[:, 0:num_halo, num_halo:-num_halo])
    t_rcvbuf = np.empty_like(field[:, -num_halo:, num_halo:-num_halo])
    reqs_tb = []
    reqs_tb.append(p.comm().Irecv(b_rcvbuf, source = p.bottom()))
    reqs_tb.append(p.comm().Irecv(t_rcvbuf, source = p.top()))

    # allocate recv buffers and pre-post the receives (left and right edge, including corners)
    l_rcvbuf = np.empty_like(field[:, :, 0:num_halo])
    r_rcvbuf = np.empty_like(field[:, :, -num_halo:])
    reqs_lr = []
    reqs_lr.append(p.comm().Irecv(l_rcvbuf, source = p.left()))
    reqs_lr.append(p.comm().Irecv(r_rcvbuf, source = p.right()))

    # pack and send (top and bottom edge, without corners)
    b_sndbuf = field[:, -2 * num_halo:-num_halo, num_halo:-num_halo].copy()
    reqs_tb.append(p.comm().Isend(b_sndbuf, dest = p.top()))
    t_sndbuf = field[:, num_halo:2 * num_halo, num_halo:-num_halo].copy()
    reqs_tb.append(p.comm().Isend(t_sndbuf, dest = p.bottom()))
    
    # wait and unpack
    for req in reqs_tb:
        req.wait()
    field[:, 0:num_halo, num_halo:-num_halo] = b_rcvbuf
    field[:, -num_halo:, num_halo:-num_halo] = t_rcvbuf
    
    # pack and send (left and right edge, including corners)
    l_sndbuf = field[:, :, -2 * num_halo:-num_halo].copy()
    reqs_lr.append(p.comm().Isend(l_sndbuf, dest = p.right()))
    r_sndbuf = field[:, :, num_halo:2 * num_halo].copy()
    reqs_lr.append(p.comm().Isend(r_sndbuf, dest = p.left()))

    # wait and unpack
    for req in reqs_lr:
        req.wait()
    field[:, :, 0:num_halo] = l_rcvbuf
    field[:, :, -num_halo:] = r_rcvbuf
            

def apply_diffusion( in_field, out_field, alpha, num_halo, num_iter=1, p=None ):
    """Integrate 4th-order diffusion equation by a certain number of iterations.
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    alpha     -- diffusion coefficient (dimensionless)
    
    Keyword arguments:
    num_iter  -- number of iterations to execute
    """

    tmp_field = np.empty_like( in_field )
    
    for n in range(num_iter):
        
        update_halo( in_field, num_halo, p )
        
        laplacian( in_field, tmp_field, num_halo=num_halo, extend=1 )
        laplacian( tmp_field, out_field, num_halo=num_halo, extend=0 )
        
        out_field[:, num_halo:-num_halo, num_halo:-num_halo] = \
            in_field[:, num_halo:-num_halo, num_halo:-num_halo] \
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]

        if n < num_iter - 1:
            in_field, out_field = out_field, in_field

            
@click.command()
@click.option('--nx', type=int, required=True, help='Number of gridpoints in x-direction')
@click.option('--ny', type=int, required=True, help='Number of gridpoints in y-direction')
@click.option('--nz', type=int, required=True, help='Number of gridpoints in z-direction')
@click.option('--num_iter', type=int, required=True, help='Number of iterations')
@click.option('--num_halo', type=int, default=2, help='Number of halo-pointers in x- and y-direction')
@click.option('--plot_result', type=bool, default=False, help='Make a plot of the result?')
def main(nx, ny, nz, num_iter, num_halo=2, plot_result=False):
    """Driver for apply_diffusion that sets up fields and does timings"""
    
    assert 0 < nx <= 1024*1024, 'You have to specify a reasonable value for nx'
    assert 0 < ny <= 1024*1024, 'You have to specify a reasonable value for ny'
    assert 0 < nz <= 1024, 'You have to specify a reasonable value for nz'
    assert 0 < num_iter <= 1024*1024, 'You have to specify a reasonable value for num_iter'
    assert 0 < num_halo <= 256, 'Your have to specify a reasonable number of halo points'
    alpha = 1./32.
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    p = Partitioner(comm, [nz, ny, nx], num_halo)
    
    if rank == 0:
        f = np.zeros( (nz, ny + 2 * num_halo, nx + 2 * num_halo) )
        f[nz // 4:3 * nz // 4, num_halo + ny // 4:num_halo + 3 * ny // 4, num_halo + nx // 4:num_halo + 3 * nx // 4] = 1.0
    else:
        f = np.empty(1)
    in_field = p.scatter(f)
    
    out_field = np.copy( in_field )

    f = p.gather(in_field)
    if rank == 0:
        np.save('in_field', f)
        if plot_result:
            plt.ioff()
            plt.imshow(f[in_field.shape[0] // 2, :, :], origin='lower')
            plt.colorbar()
            plt.savefig('in_field.png')
            plt.close()
    
    # warmup caches
    apply_diffusion( in_field, out_field, alpha, num_halo, p=p )

    comm.Barrier()
    
    # time the actual work
    tic = time.time()
    apply_diffusion( in_field, out_field, alpha, num_halo, num_iter=num_iter, p=p )
    toc = time.time()
    
    comm.Barrier()
    
    if rank == 0:
        print("Elapsed time for work = {} s".format(toc - tic) )

    update_halo(out_field, num_halo, p)

    f = p.gather(out_field)
    if rank == 0:
        np.save('out_field', f)
        if plot_result:
            plt.imshow(f[out_field.shape[0] // 2, :, :], origin='lower')
            plt.colorbar()
            plt.savefig('out_field.png')
            plt.close()


if __name__ == '__main__':
    main()
    



# ******************************************************
#     Program: stencil3d
#     Authors: Oliver Fuhrer, Beat Hubmann, Shruti Nath
#        Date: June-August 2020
# Description: Simple stencil example on a cubed sphere
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
from cubedspherepartitioner import CubedSpherePartitioner


def laplacian( in_field, lap_field, num_halo, extend=0 ):
    """Compute Laplacian using 2nd-order centered differences.
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    num_halo  -- number of halo points
    
    Keyword arguments:
    extend    -- extend computation into halo-zone by this number of points (either 0 or 1).
                 this also uses non-corner halo data to correct stencil for missing corner halo data.
    """

    assert -1 < extend < 2, 'Can only extend Laplacian calculation up to 1 point into halo'

    ib = num_halo - extend
    ie = - num_halo + extend
    jb = num_halo - extend
    je = - num_halo + extend
    
    lap_field[:, jb:je, ib:ie] = - 4. * in_field[:, jb    :je    , ib    :ie]  \
                                      + in_field[:, jb    :je    , ib - 1:ie - 1] \
                                      + in_field[:, jb    :je    , ib + 1:ie + 1 if ie != -1 else None]  \
                                      + in_field[:, jb - 1:je - 1, ib:ie] \
                                      + in_field[:, jb + 1:je + 1 if je != -1 else None, ib:ie] 

    # fix missing stencil info from zero-ed halo corners:
    if extend > 0:
        # bottom-left:
        lap_field[:, jb, num_halo] += in_field[:, num_halo, ib] # lap_field(-1, 0) += in_field( 0,-1)
        lap_field[:, num_halo, ib] += in_field[:, jb, num_halo] # lap_field( 0,-1) += in_field(-1, 0)
        # bottom-right:
        lap_field[:, jb, -num_halo] += in_field[:, num_halo, ie] # lap_field(-1, nx-1) += in_field( 0, nx  ) 
        lap_field[:, num_halo, ie] += in_field[:, jb, -num_halo] # lap_field( 0, nx  ) += in_field(-1, nx-1)  
        # top-left:
        lap_field[:, je, num_halo] += in_field[:, -num_halo, ib] # lap_field(ny  , 0) += in_field(ny-1,-1) 
        lap_field[:, -num_halo, ib] += in_field[:, je, num_halo] # lap_field(ny-1,-1) += in_field(ny  , 0)  
        # top-right:
        lap_field[:, je, -num_halo] += in_field[:, -num_halo, ie] # lap_field(ny  , nx-1) += in_field(ny-1, nx  )  
        lap_field[:, -num_halo, ie] += in_field[:, je, -num_halo] # lap_field(ny-1, nx  ) += in_field(ny  , nx-1)


def update_halo( field, num_halo, p=None ):
    """Update the halo-zone. 
    
    field    -- input/output field (nz x ny x nx with halo in x- and y-direction)
    num_halo -- number of halo points
    """

    reqs_recv, reqs_send = [], []

    # allocate recv buffers and pre-post the receives (top and bottom edge, without corners):
    b_rcvbuf = np.empty_like(field[:, 0:num_halo, num_halo:-num_halo])
    reqs_recv.append(p.comm().Irecv(b_rcvbuf, source = p.bottom()))
    t_rcvbuf = np.empty_like(field[:, -num_halo:, num_halo:-num_halo])
    reqs_recv.append(p.comm().Irecv(t_rcvbuf, source = p.top()))
    # allocate recv buffers and pre-post the receives (left and right edge, without corners):
    l_rcvbuf = np.empty_like(field[:, num_halo:-num_halo:, 0:num_halo])
    reqs_recv.append(p.comm().Irecv(l_rcvbuf, source = p.left()))
    r_rcvbuf = np.empty_like(field[:, num_halo:-num_halo, -num_halo:])
    reqs_recv.append(p.comm().Irecv(r_rcvbuf, source = p.right()))

    # rotate to fit receiver's halo orientation, then
    # pack and send (top and bottom edge, without corners):
    b_sndbuf = np.rot90(field[:, num_halo:2 * num_halo, num_halo:-num_halo],
                        p.rot_halo_bottom(),
                        axes=(1,2)).copy()
    reqs_send.append(p.comm().Isend(b_sndbuf, dest = p.bottom()))
    t_sndbuf = np.rot90(field[:, -2 * num_halo:-num_halo, num_halo:-num_halo],
                        p.rot_halo_top(),
                        axes=(1,2)).copy()
    reqs_send.append(p.comm().Isend(t_sndbuf, dest = p.top()))

    # rotate to fit receiver's halo orientation, then
    # pack and send (left and right edge, without corners):
    l_sndbuf = np.rot90(field[:, num_halo:-num_halo, num_halo:2 * num_halo],
                        p.rot_halo_left(),
                        axes=(1,2)).copy()
    reqs_send.append(p.comm().Isend(l_sndbuf, dest = p.left()))    
    r_sndbuf = np.rot90(field[:, num_halo:-num_halo, -2 * num_halo:-num_halo],
                        p.rot_halo_right(),
                        axes=(1,2)).copy()
    reqs_send.append(p.comm().Isend(r_sndbuf, dest = p.right()))
       
    # wait and unpack:
    for req in reqs_recv:
        req.wait()
    
    field[:,         0: num_halo, num_halo:-num_halo] = b_rcvbuf
    field[:, -num_halo:         , num_halo:-num_halo] = t_rcvbuf
    field[:,  num_halo:-num_halo,        0: num_halo] = l_rcvbuf
    field[:,  num_halo:-num_halo,-num_halo:         ] = r_rcvbuf

    # wait for sends to complete:
    for req in reqs_send:
        req.wait()
            

def apply_diffusion( in_field, out_field, alpha, num_halo, num_iter=1, p=None, smoothing=True ):
    """Integrate 4th-order diffusion equation by a certain number of iterations.
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    alpha     -- diffusion coefficient (dimensionless)
    
    Keyword arguments:
    num_iter  -- number of iterations to execute
    p -- responsible partitioner instance 
    smoothing -- fix halo corner points using the three surrounding field points - unstable if False
    """

    tmp_field = np.zeros_like( in_field )
    
    for n in range(num_iter):
        
        update_halo( in_field, num_halo, p )

        if smoothing:
            # apply smoothing filter to corner halo points to dampen numerical corner errors:
            jcb = icb = num_halo
            jce = ice = -num_halo - 1
            # bottom-left:
            avg = ( in_field[:, jcb  , icb  ] \
                    + in_field[:, jcb  , icb-1] \
                    + in_field[:, jcb-1, icb  ] ) / 3.
            in_field[:, jcb-1, icb-1] = avg
            # bottom-right:
            avg = ( in_field[:, jcb  , ice  ] \
                    + in_field[:, jcb  , ice+1] \
                    + in_field[:, jcb-1, ice  ] ) / 3.
            in_field[:, jcb-1, ice+1] = avg
            # top-left:
            avg = ( in_field[:, jce  , icb  ] \
                    + in_field[:, jce  , icb-1] \
                    + in_field[:, jce+1, icb  ] ) / 3.
            in_field[:, jce+1, icb-1] = avg
            # top-right:
            avg = ( in_field[:, jce  , ice  ] \
                    + in_field[:, jce  , ice+1] \
                    + in_field[:, jce+1, ice  ] ) / 3.
            in_field[:, jce+1, ice+1] = avg 

        laplacian( in_field, tmp_field, num_halo=num_halo, extend=1 )
        laplacian( tmp_field, out_field, num_halo=num_halo, extend=0 )

        out_field[:, num_halo:-num_halo, num_halo:-num_halo] = \
            in_field[:, num_halo:-num_halo, num_halo:-num_halo] \
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]

        if n < num_iter - 1:
            in_field, out_field = out_field, in_field


def check_halo(halo, neighbor, tolerance=1e-4):
    """Checks if 1) halo originates from neighbor
                 2) halo does not contain any zero corner field elements indicating a faulty shift
                 3) halo is properly oriented along increasing y-axis
                 4) halo is properly oriented along increasing x-axis""" 
    halo_int = halo.astype(int)
    return np.all(np.isclose(halo - neighbor * 1e-3, halo_int, tolerance)) and \
           np.all(halo_int > 0) and \
           np.all(np.diff(halo, axis=1) > 0) and \
           np.all(np.diff(halo, axis=2) > 0)


def verify_halo_exchange(field, rank, local_rank, tile, num_halo, p):
    """Checks halo correctness after initial halo exchange during verification"""
    # bottom halo:
    bottom_halo = np.rot90(field[:, :num_halo, num_halo:-num_halo],
                  -p.rot_halo_bottom(),
                  axes=(2,1))
    assert check_halo(bottom_halo, p.bottom()), 'Bottom halo exchange on tile {}, rank {} faulty'.format(tile, rank)
    # top halo:
    top_halo = np.rot90(field[:, -num_halo:, num_halo:-num_halo],
               -p.rot_halo_top(),
               axes=(2,1))
    assert check_halo(top_halo, p.top()), 'Top halo exchange on tile {}, rank {} faulty'.format(tile, rank)
    # left halo:
    left_halo = np.rot90(field[:, num_halo:-num_halo, :num_halo],
                -p.rot_halo_left(),
                axes=(2,1))
    assert check_halo(left_halo, p.left()), 'Left halo exchange on tile {}, rank {} faulty'.format(tile, rank)
    # right halo:
    right_halo = np.rot90(field[:, num_halo:-num_halo, -num_halo:],
                 -p.rot_halo_right(),
                 axes=(2,1))
    assert check_halo(right_halo, p.right()), 'Right halo exchange on tile {}, rank {} faulty'.format(tile, rank)

    # write to standard output for visual diagnostics if fields are small enough:
    if field.shape[1] <= 12:
        with np.printoptions(precision=3, suppress=True, linewidth=120):
            print("global rank {}, local rank {}, tile {}: subtile after one halo exchange:\n{}\n".format(rank,
                local_rank, p.tile(), np.flipud(field[0,:,:])))
    

@click.command()
@click.option('--nx', type=int, required=True, help='Number of gridpoints in x-direction')
@click.option('--ny', type=int, required=True, help='Number of gridpoints in y-direction')
@click.option('--nz', type=int, required=True, help='Number of gridpoints in z-direction')
@click.option('--num_iter', type=int, required=True, help='Number of iterations')
@click.option('--num_halo', type=int, default=2, help='Number of halo-pointers in x- and y-direction')
@click.option('--plot_result', type=bool, default=False, help='Make a plot of the result?')
@click.option('--verify', type=bool, default=False, help='Output verification plots? No diffusion, overrides num_iter, plot_result options')
def main(nx, ny, nz, num_iter, num_halo=2, plot_result=False, verify=False):
    """Driver for apply_diffusion that sets up fields and does timings"""
    
    assert 0 < nx <= 1024*1024, 'You have to specify a reasonable value for nx'
    assert 0 < ny <= 1024*1024, 'You have to specify a reasonable value for ny'
    assert 0 < nz <= 1024, 'You have to specify a reasonable value for nz'
    assert 0 < num_iter <= 1024*1024, 'You have to specify a reasonable value for num_iter'
    assert 0 < num_halo <= 256, 'Your have to specify a reasonable number of halo points'
    alpha = 1./32.
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    p = CubedSpherePartitioner(comm, [nz, ny, nx], num_halo)

    tile = p.tile()
    local_rank = p.local_rank()

    if local_rank == 0:
        f = np.zeros( (nz, ny + 2 * num_halo, nx + 2 * num_halo) )
        if not verify:
            # choose initial pattern to be diffused:
            
            # option 1: Like stencil2d-mpi during HPC4WC course:
            # f[nz // 4:3 * nz // 4, num_halo + ny // 4:num_halo + 3 * ny // 4, num_halo + nx // 4:num_halo + 3 * nx // 4] = 1.0
            
            # option 2: Similar to option 1, but positive region extended towards tile edges:
            # f[nz // 10:9 * nz // 10, num_halo + ny // 10:num_halo + 9 * ny // 10, num_halo + nx // 10:num_halo + 9 * nx // 10] = 1.0

            # option 3: One positive region in bottom-left (0-0) corner, one positive region in top-right (ny-nx) corner:       
            # f[nz // 4:3 * nz // 4, num_halo:num_halo + ny // 4, num_halo:num_halo + nx // 4] = 1.0
            # f[nz // 4:3 * nz // 4, num_halo + 3 * ny // 4:-num_halo, num_halo + 3 * nx // 4:-num_halo] = 1.0

            # option 4: Positive region line prime number fraction off-center across tile:
            f[nz // 4:3 * nz // 4, num_halo + ny // 7:num_halo + 2 * ny // 7, num_halo:-num_halo] = 1.0

            # option 5: Similar to option 3, but positive region value is rank- and position-flavored:
            # f[nz // 4:3 * nz // 4, num_halo:num_halo + ny // 4, num_halo:num_halo + nx // 4] = rank * 100 + 125
            # f[nz // 4:3 * nz // 4, num_halo + 3 * ny // 4:-num_halo, num_halo + 3 * nx // 4:-num_halo] = rank * 100 + 175
    else:
        f = np.empty(1)
    in_field = p.scatter(f)

    # add pattern increasing in steps of 100 in pos y-direction, steps of 1 in pos x-direction with
    # rank encoded in the 3 decimal places right of the decimal separator:
    if verify:
        local_ny, local_nx = in_field.shape[1:]
        test_grid = np.add(*np.mgrid[100:(local_ny - 2 * num_halo + 1) * 100:100, 1:(local_nx - 2 * num_halo + 1)]) + rank * 1e-3 
        in_field[:, num_halo:-num_halo, num_halo:-num_halo] = test_grid	

    out_field = np.copy( in_field )

    if plot_result or verify:
        np.save('local_in_field_{}-{}_{}{}'.format(tile, local_rank, rank, '_verify' if verify else ''), in_field)
        plt.ioff()
        plt.imshow(in_field[in_field.shape[0] // 2, :, :], origin='lower')
        plt.colorbar()
        plt.savefig('local_in_field_{}-{}_{}{}'.format(tile, local_rank, rank, '_verify' if verify else ''))
        plt.close()

    # no diffusion performed for verification:
    if not verify:
        # warmup caches:
        apply_diffusion( in_field, out_field, alpha, num_halo, p=p )

        comm.Barrier()
        # time the actual work: -------------------------------------------------------
        tic = time.time()
        apply_diffusion( in_field, out_field, alpha, num_halo, num_iter=num_iter, p=p )
        toc = time.time()
        # -----------------------------------------------------------------------------
        comm.Barrier()
        
        if rank == 0:
            print("nz = {:>3}, ny = {:>3}, nx = {:>3}; iter = {:>4}, ranks = {:>3} // Elapsed time for work = {:>9.4f} s".format(nz, ny, nx,
                num_iter, comm.Get_size(), toc - tic))

    update_halo(out_field, num_halo, p)

    # halo exchange correctness verification after initial update_halo():
    if verify:
        verify_halo_exchange(out_field, rank, local_rank, tile, num_halo, p)
        if rank == 0:
            print(80 * '*' + '\n Verification complete - all halo exchange tests passed!\n' + 80 * '*')

    if plot_result or verify:
        np.save('local_out_field_{}-{}_{}{}'.format(tile, local_rank, rank, '_verify' if verify else ''),
                out_field[:, num_halo:-num_halo, num_halo:-num_halo])
        plt.ioff()
        plt.imshow(out_field[out_field.shape[0] // 2, num_halo:-num_halo, num_halo:-num_halo], origin='lower')
        plt.colorbar()
        plt.savefig('local_out_field_{}-{}_{}{}'.format(tile, local_rank, rank, '_verify' if verify else ''))
        plt.close()


if __name__ == '__main__':
    main()
    



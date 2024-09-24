# ******************************************************
#     Program: stencil2d-cupy
#      Author: Stefano Ubbiali, Oliver Fuhrer
#       Email: subbiali@phys.ethz.ch, ofuhrer@ethz.ch
#        Date: 04.06.2020
# Description: CuPy implementation of 4th-order diffusion
# ******************************************************
import time
import click
import numpy as np
import cupy as cp


def boundary_update(u, bdry):

    # South boundary (without corners):
    u[:, :bdry, bdry:-bdry] = u[:, -2*bdry:-bdry, bdry:-bdry]

    # North boundary (without corners):
    u[:, -bdry:, bdry:-bdry] = u[:, bdry:2*bdry, bdry:-bdry]

    # West boundary (including corners):
    u[:, :, :bdry] = u[:, :, -2*bdry:-bdry]

    # East boundary (including corners):
    u[:, :, -bdry:] = u[:, :, bdry:2*bdry]


def apply_diffusion(u, alpha, bdry, itrs):

    lap = cp.empty_like(u)

    for _ in range(itrs):
        # Boundary update:
        boundary_update(u, bdry)

        # Run the stencil:
        imin = bdry - 1
        imax = -bdry + 1
        jmin = bdry - 1
        jmax = -bdry + 1

        lap[:, jmin:jmax, imin:imax] = (
             -4 * u[:, jmin:jmax, imin:imax]
                + u[:, jmin:jmax, imin-1:imax-1]
                + u[:, jmin:jmax, imin+1:imax+1 if imax != -1 else None]
                + u[:, jmin-1:jmax-1, imin:imax]
                + u[:, jmin+1:jmax+1 if jmax != -1 else None, imin:imax])

        imin = bdry
        imax = -bdry
        jmin = bdry
        jmax = -bdry

        u[:, jmin:jmax, imin:imax] -= alpha * (
             -4 * lap[:, jmin:jmax, imin:imax]
                + lap[:, jmin:jmax, imin-1:imax-1]
                + lap[:, jmin:jmax, imin+1:imax+1]
                + lap[:, jmin-1:jmax-1, imin:imax]
                + lap[:, jmin+1:jmax+1, imin:imax])

    boundary_update(u, bdry)


@click.command()
@click.option('-nx', type=int, required=True, help='Number of gridpoints in x-direction')
@click.option('-ny', type=int, required=True, help='Number of gridpoints in y-direction')
@click.option('-nz', type=int, required=True, help='Number of gridpoints in z-direction')
@click.option('-itrs', type=int, required=True, help='Number of iterations')
@click.option('-bdry', type=int, default=2, help='Number of boundary points in x- and y-direction')

def main(nx=128, ny=128, nz=64, itrs=1024, bdry=2):
    """Driver for apply_diffusion that sets up fields and does timings"""

    assert 0 < nx <= 1024 * 1024, 'You have to specify a reasonable value for nx'
    assert 0 < ny <= 1024 * 1024, 'You have to specify a reasonable value for ny'
    assert 0 < nz <= 1024, 'You have to specify a reasonable value for nz'
    assert 0 < itrs <= 1024 * 1024, 'You have to specify a reasonable value for itrs'
    assert 2 <= bdry <= 256, 'You have to specify a reasonable number of boundary points'

    alpha = 1 / 32

    # Allocate input field:
    xsize = nx + 2 * bdry
    ysize = ny + 2 * bdry
    zsize = nz

    u_host = np.zeros((zsize, ysize, xsize))

    # Prepare input field:
    imin = int(0.25 * xsize + 0.5)
    imax = int(0.75 * xsize + 0.5)
    jmin = int(0.25 * ysize + 0.5)
    jmax = int(0.75 * ysize + 0.5)

    u_host[:, jmin:jmax+1, imin:imax+1] = 1

    # Write input field to file:
    np.save('in_field', u_host)

    # Timed region:
    tic = time.time()

    u = cp.array(u_host)

    apply_diffusion(u, alpha, bdry, itrs=itrs)

    u_host = u.get()

    toc = time.time()
    print(f'Elapsed time for work = {toc-tic:.16f}s.')

    # Save output field:
    np.save('out_field', u_host)


if __name__ == '__main__':
    main()

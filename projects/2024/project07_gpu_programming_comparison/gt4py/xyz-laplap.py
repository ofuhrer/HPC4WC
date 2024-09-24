# ******************************************************
#     Program: stencil2d-gt4py
#      Author: Stefano Ubbiali
#       Email: subbiali@phys.ethz.ch
#        Date: 04.06.2020
# Description: GT4Py implementation of 4th-order diffusion
# ******************************************************
import time
import click
import numpy as np
import gt4py as gt

try:
    # Modern GT4Py (as is available on PyPI):
    import gt4py.cartesian.gtscript as gtscript
    legacy_api = False
except ImportError:
    # Ancient GT4Py (as is installed on Piz Daint):
    import gt4py.gtscript as gtscript
    legacy_api = True


@gtscript.function
def laplacian(u):
    lap = (
     -4 * u[0, 0, 0]
        + u[-1, 0, 0]
        + u[1, 0, 0]
        + u[0, -1, 0]
        + u[0, 1, 0])
    return lap


def diffusion_defs(in_field: gtscript.Field[float], out_field: gtscript.Field[float], *, alpha: float):

    from __externals__ import laplacian
    from __gtscript__ import PARALLEL, computation, interval

    with computation(PARALLEL), interval(...):
        lap = laplacian(in_field)
        laplap = laplacian(lap)
        out_field = in_field - alpha * laplap


def boundary_update(u, bdry):
    # South boundary (without corners):
    u[bdry:-bdry, :bdry, :] = u[bdry:-bdry, -2*bdry:-bdry, :]

    # North boundary (without corners):
    u[bdry:-bdry, -bdry:, :] = u[bdry:-bdry, bdry:2*bdry, :]

    # West boundary (including corners):
    u[:bdry, :, :] = u[-2*bdry:-bdry, :, :]

    # East boundary (including corners):
    u[-bdry:, :, :] = u[bdry:2*bdry, :, :]


def apply_diffusion(diffusion_stencil, u, v, alpha, bdry, itrs):

    # Origin and extent of the computational domain:
    origin = (bdry, bdry, 0)
    domain = (u.shape[0]-2*bdry, u.shape[1]-2*bdry, u.shape[2])

    for _ in range(itrs // 2):
        # Boundary update:
        boundary_update(u, bdry)

        # Run the stencil:
        diffusion_stencil(in_field=u, out_field=v, alpha=alpha, origin=origin, domain=domain)

        # Boundary update:
        boundary_update(v, bdry)

        # Run the stencil:
        diffusion_stencil(in_field=v, out_field=u, alpha=alpha, origin=origin, domain=domain)

    # Boundary update:
    boundary_update(u, bdry)

    if itrs % 2 == 1:
        # Run the stencil:
        diffusion_stencil(in_field=u, out_field=v, alpha=alpha, origin=origin, domain=domain)

        # Boundary update:
        boundary_update(v, bdry)

        # East boundary update (for some reason necessary to make results match with CuPy;
        #                       only necessary for the GT4Py version installed on Piz Daint):
        v[-bdry:, :, :] = v[bdry:2*bdry, :, :]

    else:
        # East boundary update (for some reason necessary to make results match with CuPy;
        #                       only necessary for the GT4Py version installed on Piz Daint):
        u[-bdry:, :, :] = u[bdry:2*bdry, :, :]


@click.command()
@click.option('-nx', type=int, required=True, help='Number of gridpoints in x-direction')
@click.option('-ny', type=int, required=True, help='Number of gridpoints in y-direction')
@click.option('-nz', type=int, required=True, help='Number of gridpoints in z-direction')
@click.option('-itrs', type=int, required=True, help='Number of iterations')
@click.option('-bdry', type=int, default=2, help='Number of boundary points in x- and y-direction')
@click.option('-bknd', type=str, required=False, default='cuda', help='GT4Py backend')

def main(nx=128, ny=128, nz=64, itrs=1024, bdry=2, bknd='cuda'):
    """Driver for apply_diffusion that sets up fields and does timings."""

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

    u_host = np.zeros((xsize, ysize, zsize), dtype=float)

    # Prepare input field:
    imin = int(0.25 * xsize + 0.5)
    imax = int(0.75 * xsize + 0.5)
    jmin = int(0.25 * ysize + 0.5)
    jmax = int(0.75 * ysize + 0.5)

    u_host[imin:imax+1, jmin:jmax+1, :] = 1

    # Write input field to file:
    np.save('in_field', np.swapaxes(u_host, 0, 2))

    # Compile diffusion stencil:
    diffusion_stencil = gtscript.stencil(
        definition=diffusion_defs,
        backend=bknd,
        externals={'laplacian': laplacian},
        rebuild=False)

    # Timed region:
    tic = time.time()

    if legacy_api:
        u = gt.storage.from_array(backend=bknd, default_origin=(bdry, bdry, 0), data=u_host)
        v = gt.storage.empty(backend=bknd, default_origin=(bdry, bdry, 0),
                             shape=(nx+2*bdry, ny+2*bdry, nz), dtype=float)
    else:
        u = gt.storage.from_array(backend=bknd, data=u_host)
        v = gt.storage.empty(backend=bknd, shape=(nx+2*bdry, ny+2*bdry, nz), dtype=float)

    apply_diffusion(diffusion_stencil, u, v, alpha, bdry, itrs)

    if legacy_api:
        u_host = np.asarray(u if itrs % 2 == 0 else v)
    else:
        u_host = (u if itrs % 2 == 0 else v).get()

    toc = time.time()
    print(f'Elapsed time for work = {toc-tic:.16f}s.')

    # Save output field:
    np.save('out_field', np.swapaxes(u_host, 0, 2))


if __name__ == '__main__':
    main()

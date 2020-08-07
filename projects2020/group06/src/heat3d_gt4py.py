# operator for 3d heat equation using gt4py

import time
import numpy as np
import cupy as cp
import gt4py as gt
from gt4py import gtscript
import matplotlib.pyplot as plt
import click
from sync import get_time

@gtscript.function
def laplace_gt4py_3D(phi, dx, dy, dz):
    '''
    Laplace operator in 3 dimensions
    
    phi -- field onto which the operator should be applied
    dx, dy, dz -- spacing in each direction
    '''
    lap = ( ( -2 * phi[ 0, 0, 0] + phi[1, 0, 0] + phi[-1, 0, 0])/dx**2 + 
            ( -2 * phi[ 0, 0, 0] + phi[0, 1, 0] + phi[ 0,-1, 0])/dy**2 + 
            ( -2 * phi[ 0, 0, 0] + phi[0, 0, 1] + phi[ 0, 0,-1])/dz**2 )
    return lap

def heat_gt4py_3D(u_now: gtscript.Field[np.float64], u_next: gtscript.Field[np.float64],
                     *, alpha: np.float64, dt: np.float64, dx: np.float64, dy: np.float64, dz: np.float64):
    '''
    Definiton for the stencil in gt4py
    
    u_now -- field at the time t, as gtscript Field
    u_next -- field at the time t+dt, as gtscript Field
    alpha -- thermal diffusivity
    dt -- time step
    dx, dy, dz -- spacing in each direction
    '''
    
    from __externals__ import laplace_gt4py_3D
    from __gtscript__ import PARALLEL, computation, interval
    
    with computation(PARALLEL):
        with interval(1,-1):
            u_next = u_now + dt*alpha*laplace_gt4py_3D(u_now, dx, dy, dz)
            

def apply_heat_stencil(u_now, u_next, alpha, Tcool, dx, dy, dz, dt, nt, stencil, domain, origin, num_border, backend):
    '''
    Time integration for the heat equation
    
    u_now -- field at the time t, as gtscript Field
    u_next -- field at the time t+nt*dt, as gtscript Field
    alpha -- thermal diffusivity
    Tcool -- boundary temperature
    dx, dy, dz -- spacing in each direction
    dt -- time step
    nt -- number of time steps
    stencil -- the stencil in compiled by gt4py
    domain -- domain in onto which stencil should be applied
    origin -- origin for the stencil
    num_border -- number of gridpoints which are borders
    backend -- the device on which the computation should be done, i.e. the backend to be used
    '''
    
    for n in range(nt):
        # Boundary Condition Update - first define bv apropriate to your chosen backend
        if backend == 'gtcuda':
            bv = cp.asarray(Tcool)
        else:
            bv = Tcool
        
        u_now[:,0:(num_border),:] = bv
        u_now[:,-(num_border):,:] = bv
        u_now[0:(num_border),:,:] = bv
        u_now[-(num_border):,:,:] = bv
        u_now[:,:,0:(num_border)] = bv
        u_now[:,:,-(num_border):] = bv
        
        #run the stencil
        stencil(
            u_now=u_now,
            u_next=u_next,
            alpha=alpha,
            dt = dt,
            dx = dx,
            dy = dy,
            dz = dz,
            domain = domain,
            origin = origin
        )

        if n < nt - 1:
            u_next, u_now = u_now, u_next

    return u_next

def heat3d_gt4py(in_field, alpha, Tcool, dx, dy, dz, dt, nt, backend='numpy', result='time', num_border=1):
    '''
    Solves the 3 dimensional heat equation using gt4py.
    
    in_field  -- input field (nx x ny x nz)
    alpha -- thermal diffusivity
    Tcool -- initial temperature
    dx, dy, dz -- spacing in x direction
    dt -- time step
    nt -- number of iterations
    backend -- the backend for gt4py on which the equation is to be solved
    result -- either 'time', 'field' or 'both', returning either the total time the computation took, the resulting field or both 
    '''
    dx = 2. / (in_field.shape[0] - 1)
    dy = 2. / (in_field.shape[1] - 1)
    dz = 2. / (in_field.shape[2] - 1)
    
    origin = (num_border,num_border,0)
    domain = (
        in_field.shape[0] - 2 * num_border,
        in_field.shape[1] - 2 * num_border,
        in_field.shape[2]
    )
    
    # create fields in gtstorage
    old_field_gt4py = gt.storage.from_array(in_field, backend, origin, dtype=np.float64)
    new_field_gt4py = gt.storage.empty(backend, origin, in_field.shape, dtype=np.float64)
    
    # create stencil
    heat_stencil = gtscript.stencil(
        definition=heat_gt4py_3D, 
        backend=backend, 
        externals={"laplace_gt4py_3D": laplace_gt4py_3D}
    )
    
    tic = get_time()
    out_field = apply_heat_stencil(u_now = old_field_gt4py, u_next = new_field_gt4py, alpha = alpha, Tcool = Tcool, 
                                   dx = dx, dy = dy, dz = dz, dt = dt, nt = nt, stencil = heat_stencil, 
                                   domain = domain, origin = origin, num_border = num_border, backend = backend)
    time_gt4py = get_time() - tic
    
    if result == 'time':
        return time_gt4py
    elif result == 'field':
        return out_field
    elif result == 'both':
        return time_gt4py, out_field
    
@click.command()
@click.option('--nx', type=int, required=True, help='Number of gridpoints in x-direction')
@click.option('--ny', type=int, required=True, help='Number of gridpoints in y-direction')
@click.option('--nz', type=int, required=True, help='Number of gridpoints in z-direction')
@click.option('--nt', type=int, required=True, help='Number of timesteps')
@click.option('--plot_result', type=bool, default=False, help='Make a plot of the result?')
def main(nx, ny, nz, nt, plot_result=False):
    '''
    Solves the use case defined in report.
    
    nx, ny, nz -- number of grid points in each dimension. Can't neither be smaller than zero nor exorbitantly large
    nt -- number of iterations. Can't neither be smaller than zero nor exorbitantly large
    plot_result -- specifies wheter results should be plotet or not
    '''
    assert 0 < nx <= 1024*1024, 'You have to specify a reasonable value for nx'
    assert 0 < ny <= 1024*1024, 'You have to specify a reasonable value for ny'
    assert 0 < nz <= 1024*1024, 'You have to specify a reasonable value for nz'
    assert 0 < nt <= 1024*1024, 'You have to specify a reasonable value for nt'
    
    alpha = 19.
    Tcool, Thot = 300., 400.

    x = y = z = 2.
    dx = dy = dz = x / (nx - 1)
    dt = alpha/10000 * (1/dx**2 + 1/dy**2 + 1/dz**2)**(-1)  
    
    in_field = Tcool*np.ones((nx,ny,nz))
    in_field[nx//4:3*nx//4, ny//4:3*ny//4, nz//4:3*nz//4] = Thot  

    in_field0 = np.copy(in_field)
    time_gt, out_field = heat3d_gt4py(in_field, alpha, Tcool, dx, dy, dz, dt, nt, backend='gtcuda', result='both')

    print(f"Elapsed time for work = {time_gt} s")

    np.save('out_field_gt', out_field)
    
    if plot_result:    

        linewidth=0
        
        fig = plt.figure(figsize=(9,5))

        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.set_title('initial')
        x_coord = np.linspace(0., x, in_field.shape[0])
        y_coord = np.linspace(0., y, in_field.shape[1])
        X, Y = np.meshgrid(x_coord, y_coord, indexing='ij')
        im1 = ax.plot_surface(X, Y, in_field0[:,:,in_field.shape[2] // 2], cmap=cm.viridis, rstride=1, cstride=1,
                              linewidth=linewidth, antialiased=False)

        ax = fig.add_subplot(1,2,2,projection='3d')
        ax.set_title(f'after {nt} timestemps') 
        im2 = ax.plot_surface(X, Y, out_field[:,:,out_field.shape[2] // 2], cmap=cm.viridis, rstride=1, cstride=1,
                              linewidth=linewidth, antialiased=False)
        plt.show()        
        plt.savefig('fields_gt.png')

if __name__ == '__main__':
    main()

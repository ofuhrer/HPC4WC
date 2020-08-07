# operator for 3d heat equation using cupy

import time
import numpy as np
import click
import matplotlib.pyplot as plt
from matplotlib import cm
from sync import get_time

try:
    import cupy as cp
except ImportError:
    cp = np

def op_np(in_field, out_field, alpha, Tcool, dx, dy, dz, dt, nt=1): 
    '''
    The stencil for the 3 dimensional heat equation using cupy
    
    in_field -- the field of size (nx, ny, nz) at initial time
    out_field -- the field of size (nx, ny, nz) afther the integration
    alpha -- thermal diffusivity
    Tcool -- initial cold temperature
    dx, dy, dz -- spacing in each dimension
    dt -- time step
    nt -- number of iterations
    '''
    
    # Integration
    for n in range(nt):
        
        out_field[1:-1, 1:-1, 1:-1] = (in_field[1:-1, 1:-1, 1:-1] + alpha * dt * (
                 (in_field[2:  , 1:-1, 1:-1] - 2 * in_field[1:-1, 1:-1, 1:-1] + in_field[0:-2, 1:-1, 1:-1]) / dx**2  +   
                 (in_field[1:-1, 2:  , 1:-1] - 2 * in_field[1:-1, 1:-1, 1:-1] + in_field[1:-1, 0:-2, 1:-1]) / dy**2  +
                 (in_field[1:-1, 1:-1, 2:  ] - 2 * in_field[1:-1, 1:-1, 1:-1] + in_field[1:-1, 1:-1, 0:-2]) / dz**2))

        # boundary conditions
        out_field[ 0, :, :] = Tcool
        out_field[-1, :, :] = Tcool
        out_field[ :, 0, :] = Tcool
        out_field[ :,-1, :] = Tcool
        out_field[ :, :, 0] = Tcool
        out_field[ :, :,-1] = Tcool
        
        if n < nt - 1:
            in_field, out_field = out_field, in_field

    return out_field

def heat3d_cp(alpha, Tcool, Thot, dx, dy, dz, dt, nx, ny, nz, nt, result='time'):
    '''
    Solves the 3 dimensional heat equation using cupy
    
    alpha -- thermal diffusivity
    Tcool -- initial cold temperature
    Thot -- initial hot temperature
    dx, dy, dz -- spacing in each dimension
    dt -- time step
    nx, ny, nz -- number of gridpoints in each dimension
    nt -- integration time
    result -- either 'time', 'field' or 'both', returning either the total time the computation took, the resulting field or both 
    '''
        
    in_field = Tcool*cp.ones((nx,ny,nz))
    in_field[nx//4:3*nx//4, ny//4:3*ny//4, nz//4:3*nz//4] = Thot
    
    out_field = cp.copy(in_field)

    # warm up
    op_np(in_field, out_field, alpha, Tcool, dx, dy, dz, dt)

    # timing
    tic = get_time()
    op_np(in_field, out_field, alpha, Tcool, dx, dy, dz, dt, nt=nt)
    #toc = time.time()
    time_cp = get_time() - tic
    
    if result == 'time':
        return time_cp
    elif result == 'field':
        return in_field
    elif result == 'both':
        return time_cp, in_field

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
    
    # Assert conditions on nx, ny, nz, nt
    assert 0 < nx <= 1024*1024, 'You have to specify a reasonable value for nx'
    assert 0 < ny <= 1024*1024, 'You have to specify a reasonable value for ny'
    assert 0 < nz <= 1024*1024, 'You have to specify a reasonable value for nz'
    assert 0 < nt <= 1024*1024, 'You have to specify a reasonable value for nt'
    
    # define constants and initial fields
    alpha = 19.
    Tcool, Thot = 300., 400.

    x = y = z = 2.
    dx = dy = dz = x / (nx - 1)
    dt = alpha/10000 * (1/dx**2 + 1/dy**2 + 1/dz**2)**(-1)  

    in_field = Tcool*cp.ones((nx,ny,nz))
    in_field[nx//4:3*nx//4, ny//4:3*ny//4, nz//4:3*nz//4] = Thot

    in_field0 = cp.copy(in_field)
    time_cp, out_field = heat3d_cp(alpha, Tcool, Thot, dx, dy, dz, dt, nx, ny, nz, nt, result='both')

    print(f"Elapsed time for work = {time_cp} s")

    try:
        np.save('out_field_cp', out_field.get())
    except AttributeError:
        np.save('out_field_cp', out_field)
    
    if plot_result:    

        linewidth=0
        
        fig = plt.figure(figsize=(9,5))

        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.set_title('initial')
        x_coord = np.linspace(0., x, in_field.shape[0])
        y_coord = np.linspace(0., y, in_field.shape[1])
        X, Y = np.meshgrid(x_coord, y_coord, indexing='ij')
        try:
            im1 = ax.plot_surface(X, Y, in_field0[:,:,in_field0.shape[2] // 2].get(), cmap=cm.viridis, rstride=1, cstride=1,
                              linewidth=linewidth, antialiased=False)
        except AttributeError:
            im1 = ax.plot_surface(X, Y, in_field0[:,:,in_field0.shape[2] // 2], cmap=cm.viridis, rstride=1, cstride=1,
                              linewidth=linewidth, antialiased=False)

        ax = fig.add_subplot(1,2,2,projection='3d')
        ax.set_title(f'after {nt} timestemps') 
        try:
            im2 = ax.plot_surface(X, Y, out_field[:,:,out_field.shape[2] // 2].get(), cmap=cm.viridis, rstride=1, cstride=1,
                              linewidth=linewidth, antialiased=False)
        except AttributeError:
            im2 = ax.plot_surface(X, Y, out_field[:,:,out_field.shape[2] // 2], cmap=cm.viridis, rstride=1, cstride=1,
                              linewidth=linewidth, antialiased=False)
        plt.show()        
        plt.savefig('fields_cp.png')

if __name__ == '__main__':
    main()
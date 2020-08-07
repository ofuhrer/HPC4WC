# operator for 3d heat equation using numpy

import time
import numpy as np
import click
import matplotlib.pyplot as plt
from matplotlib import cm
from sync import get_time

def op_np(in_field, out_field, alpha, Tcool, dx, dy, dz, dt, nt=1): 
    '''
    The stencil for the 3 dimensional heat equation using numpy
    
    in_field  -- input field (nx x ny x nz)
    out_field -- same size as in_field
    alpha -- thermal diffusivity
    Tcool -- initial temperature
    dx, dy, dz -- spacing in each dimension
    dt -- time step
    nt -- number of iterations
    '''
    
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

def heat3d_np(in_field, alpha, Tcool, dx, dy, dz, dt, nt, result='time'):
    '''
    Solves the 3 dimensional heat equation using numpy.
    
    in_field  -- input field (nx x ny x nz)
    alpha -- thermal diffusivity
    Tcool -- initial temperature
    dx, dy, dz -- spacing in each dimension
    dt -- time step
    nt -- number of iterations
    result -- either 'time', 'field' or 'both', returning either the total time the computation took, the resulting field or both
    '''
    
    out_field = np.copy(in_field)
    # warm up
    op_np(in_field, out_field, alpha, Tcool, dx, dy, dz, dt)

    # timing
    tic = get_time()
    op_np(in_field, out_field, alpha, Tcool, dx, dy, dz, dt, nt=nt)
    #toc = time.time()
    time_np = get_time() - tic
    
    if result == 'time':
        return time_np
    elif result == 'field':
        return in_field
    elif result == 'both':
        return time_np, in_field

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
    
    in_field = Tcool*np.ones((nx,ny,nz))
    in_field[nx//4:3*nx//4, ny//4:3*ny//4, nz//4:3*nz//4] = Thot  

    in_field0 = np.copy(in_field)
    time_np, out_field = heat3d_np(in_field, alpha, Tcool, dx, dy, dz, dt, nt, result='both')

    print(f"Elapsed time for work = {time_np} s")

    np.save('out_field_np', out_field)
    
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
        plt.savefig('fields_np.png')

if __name__ == '__main__':
    main()
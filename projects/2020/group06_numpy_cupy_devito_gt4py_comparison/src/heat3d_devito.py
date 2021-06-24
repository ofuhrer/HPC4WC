# operator for 3d heat equation using devito

from devito import SubDomain, Grid, TimeFunction, Eq, solve, Operator
import numpy as np
import time
import click
from sync import get_time

class Middle(SubDomain):
    name = 'middle'
    def define(self, dimensions):
        d = dimensions
        return {d: ('middle', 1, 1) for d in dimensions}
mid = Middle()

def heat3d_devito(in_field, alpha, Tcool, dt, nt, platform='default', result='time'): 
    '''
    Solves the 3 dimensional heat equation using devito.
    
    in_field  -- input field (nx x ny x nz)
    alpha -- thermal diffusivity
    Tcool -- initial temperature
    dt -- time step
    nt -- number of iterations
    platform -- the platform on which the equation is to be solved
    result -- either 'time', 'field' or 'both', returning either the total time the computation took, the resulting field or both 
    '''
    
    nx = in_field.shape[0]
    ny = in_field.shape[1]
    nz = in_field.shape[2]
    
    grid = Grid(shape = (nx, ny, nz), subdomains = (mid, ), extent = (2., 2., 2.))
    
    u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2, dtype=np.float64)
    u.data[:, :, :, :] = in_field    

    x, y, z = grid.dimensions
    t = grid.stepping_dim
    #boundary conditions
    bc =  [Eq(u[t+1, 0   , y, z], Tcool)]  # left
    bc += [Eq(u[t+1, nx-1, y, z], Tcool)]  # right
    bc += [Eq(u[t+1, x, ny-1, z], Tcool)]  # top
    bc += [Eq(u[t+1, x,    0, z], Tcool)]  # bottom
    bc += [Eq(u[t+1, x, y,    0], Tcool)]  # top
    bc += [Eq(u[t+1, x, y, nz-1], Tcool)]  # bottom

    #define heat equation in 3d
    eq = Eq(u.dt, alpha * ((u.dx2)+(u.dy2)+(u.dz2)))
    #solve equation
    stencil = solve(eq, u.forward)
    #create stencil
    eq_stencil = Eq(u.forward, stencil,subdomain = grid.subdomains['middle'])
    eq_stencil
    
    #solve
    if platform == 'default':
        op = Operator([eq_stencil]+bc)
    else:
        op = Operator([eq_stencil]+bc, platform=platform)
        
    #print(op.ccode)
    
    tic = get_time()
    op(time=nt, dt=dt)
    #toc = time.time()
    
    elapsed_time = get_time() - tic
    
    out_field = u.data[0,:,:,:]

    if result == 'time':
        return elapsed_time
    if result == 'field':
        return out_field
    if result == 'both':
        return elapsed_time, out_field
    
    
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
    time_devito, out_field = heat3d_devito(in_field, alpha, Tcool, dt, nt, result='both')

    print(f"Elapsed time for work = {time_devito} s")

    np.save('out_field_devito', out_field)
    
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
        plt.savefig('fields_devito.png')

if __name__ == '__main__':
    main()
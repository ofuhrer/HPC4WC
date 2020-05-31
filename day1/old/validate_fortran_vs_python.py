import time
import numpy as np
import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()


def laplacian( in_field, lap_field, num_halo, extend=0 ):
    
    # precompute indices
    ib = num_halo - extend
    ie = - num_halo + extend
    jb = num_halo - extend
    je = - num_halo + extend
    
    if ie + 1 == 0 or je + 1 == 0:
        lap_field[:, jb:je, ib:ie] = - 4. * in_field[:, jb:je, ib:ie]  \
            + in_field[:, jb:je, ib - 1:ie - 1] + in_field[:, jb:je, ib + 1:]  \
            + in_field[:, jb - 1:je - 1, ib:ie] + in_field[:, jb + 1:, ib:ie]
    else:
        lap_field[:, jb:je, ib:ie] = - 4. * in_field[:, jb:je, ib:ie]  \
            + in_field[:, jb:je, ib - 1:ie - 1] + in_field[:, jb:je, ib + 1:ie + 1]  \
            + in_field[:, jb - 1:je - 1, ib:ie] + in_field[:, jb + 1:je + 1, ib:ie]


def update_halo( field, num_halo ):
    
    # left edge (including corners)
    field[:, :, 0:num_halo] = field[:, :, -2 * num_halo:-num_halo]
    
    # right edge (including corners)
    field[:, :, -num_halo:] = field[:, :, num_halo:2 * num_halo]
    
    # bottom edge (including corners)
    field[:, 0:num_halo, :] = field[:, -2 * num_halo:-num_halo, :]
    
    # top edge (including corners)
    field[:, -num_halo:, :] = field[:, num_halo:2 * num_halo, :]
            

def work( in_field, out_field, alpha, num_halo, num_iter=1 ):
        
    tmp_field = np.empty_like( in_field )
    
    for n in range(num_iter):
        
        update_halo( in_field, num_halo )
        
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
    
    assert 0 < nx <= 1024*1024, 'You have to specify a reasonable value for nx'
    assert 0 < ny <= 1024*1024, 'You have to specify a reasonable value for ny'
    assert 0 < nz <= 1024, 'You have to specify a reasonable value for nz'
    assert 0 < num_iter <= 1024*1024, 'You have to specify a reasonable value for num_iter'
    assert 0 < num_halo <= 256, 'Your have to specify a reasonable number of halo points'
    alpha = 1./32.
    
    in_field = np.zeros( (nz, ny + 2 * num_halo, nx + 2 * num_halo) )
    for j in range(num_halo, ny + num_halo):
        for i in range(num_halo, nx + num_halo):
            in_field[:, j, i] = i % 4 + j % 2
    #in_field[:, num_halo + ny // 4:num_halo + 3 * ny // 4, num_halo + nx // 4:num_halo + 3 * nx // 4] = 1.0
    
    out_field = np.copy( in_field )
    
    # warmup caches
    work( in_field, out_field, alpha, num_halo )

    # time the actual work
    tic = time.time()
    work( in_field, out_field, alpha, num_halo, num_iter=num_iter )
    toc = time.time()
    
    print("Elapsed time for work = {} s".format(toc - tic) )
    if plot_result:
        plt.imshow(out_field[0, :, :], origin='lower')
        plt.colorbar()
        plt.savefig('result.png')
        plt.close()
        
    # validate against Fortran
    fortran_in = np.genfromtxt('fort.777', delimiter=',')
    fortran_out = np.genfromtxt('fort.778', delimiter=',')
    assert fortran_in.shape == fortran_out.shape, 'Must have same size'
    nrows, ncolumns = fortran_in.shape
    assert nrows == (nx + 2*num_halo) * (ny + 2*num_halo), 'Number of rows does not match nx * ny'
    assert ncolumns == 3, 'Must have 3 columns'
    data_in = in_field.copy()
    data_out = in_field.copy()
    for row in range(nrows):
        i = int( fortran_in[row, 0] )
        val = fortran_in[row, 1]
        j = int( fortran_in[row, 2] )
        data_in[:, j - 1, i - 1] = val
        i = int( fortran_out[row, 0] )
        val = fortran_out[row, 1]
        j = int( fortran_out[row, 2] )
        data_out[:, j - 1, i - 1] = val
    
    plt.imshow(in_field[0, :, :] - data_in[0, :, :], origin='lower')
    plt.colorbar()
    plt.savefig('diff_in.png')
    plt.close()

    plt.imshow(out_field[0, :, :], origin='lower')
    plt.colorbar()
    plt.savefig('python_out.png')
    plt.close()

    plt.imshow(data_out[0, :, :], origin='lower')
    plt.colorbar()
    plt.savefig('fortran_out.png')
    plt.close()

    plt.imshow(out_field[0, :, :] - data_out[0, :, :], origin='lower')
    plt.colorbar()
    plt.savefig('diff_out.png')
    plt.close()


if __name__ == '__main__':
    main()
    



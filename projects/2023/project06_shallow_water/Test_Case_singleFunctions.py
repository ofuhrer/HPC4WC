import gt4py as gt
from gt4py import gtscript
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np



def compute_staggered_x(
    h: gtscript.Field[float],
    hMidx: gtscript.Field[float],
    ):
    
    from __gtscript__ import PARALLEL, computation, interval
    with computation(PARALLEL), interval(...):

        # Mid-point value for h along x
            hMidx = 0.5 * (h[1,0,0] + h[0,0,0])
            
            
def compute_staggered_y(
    h: gtscript.Field[float],
    hMidy: gtscript.Field[float],
    ):
    
    from __gtscript__ import PARALLEL, computation, interval
    with computation(PARALLEL), interval(...):

        # Mid-point value for h along x
            hMidy = 0.5 * (h[0,1,0] + h[0,0,0])

############################################ AUX

def compute_aux_variables(
    h: gtscript.Field[float],
    u: gtscript.Field[float],
    v: gtscript.Field[float],
    hv: gtscript.Field[float],
    hu: gtscript.Field[float],
    ### add Ux, ... (full-field products)
    ):
    from __gtscript__ import PARALLEL, computation, interval
    with computation(PARALLEL), interval(...):
        hu = h[0,0,0]*u[0,0,0] # full grid.
        hv = h[0,0,0]*v[0,0,0] # full grid.
                
############################################ STEP 1
        
def compute_hMidx(
    h: gtscript.Field[float],
    hu: gtscript.Field[float], ### check: full-field multiplication or only staggered domain?
    hMidx_update: gtscript.Field[float],
    hMidx: gtscript.Field[float],
    ### add huMidx, hvMidx
    ):
    
    from __gtscript__ import PARALLEL, computation, interval
    with computation(PARALLEL), interval(...):
        dt = 1
        dx = 1
        
        # hu = h*u # full grid. If called here on the staggered domain, will miss the last row.
        
        # Mid-point value for h along x # staggered grid
        hMidx_update = 0.5 * dt/dx * (hu[1,0,0] - hu[0,0,0])
        hMidx = 0.5 * (h[1,0,0] + h[0,0,0]) - hMidx_update

        
def compute_hMidy(
    h: gtscript.Field[float],
    hv: gtscript.Field[float], ### check: full-field multiplication or only staggered domain?
    hMidy: gtscript.Field[float],
    ### add huMidy, hvMidy
    ):
    
    from __gtscript__ import PARALLEL, computation, interval
    with computation(PARALLEL), interval(...):
        dt = 1
        dx = 1
        
        # hu = h*u # full grid. If called here on the staggered domain, will miss the last row.
        
        # Mid-point value for h along x # staggered grid        
        hMidy = 0.5 * (h[0,1,0] + h[0,0,0]) - \
            0.5 * dt/dx * (hv[0,1,0] - hv[0,0,0])
        


############################################ STEP 2
        
def update_h(
    h: gtscript.Field[float],
    huMidx: gtscript.Field[float],
    hvMidy: gtscript.Field[float],
    ):
    #  skipping the c scaling
    from __gtscript__ import PARALLEL, computation, interval
    with computation(PARALLEL), interval(...):
        dt = 1
        dx = 1
        
        # Update fluid height ### domain shape without halo
        hnew = h[0,0,0] - \
            dt/dx * (huMidx)

        
############################################ INIT
        
def make_initial_field(shape, style='constants', do_plot=True):
    
    if style=='constants':
        h_np = np.zeros(shape)
        h_np[:5,:5,:] = 1
        h_np[:5,5:,:] = 2
        h_np[5:,5:,:] = 3
        h_np[0,:,:] = 4
        h_np[:,0,:] = 4
        h_np[:,-1,:] = 4
        h_np[-1,:,:] = 4
    elif style=='trigonometric':
        nx,ny,nz = shape
        h_np = 2.5 + (np.ones((nx,ny))*np.cos(np.linspace(0,2*np.pi,ny))).T + np.sin(np.linspace(0,2*np.pi,nx))*np.ones((nx,ny)) 
        h_np = np.expand_dims(h_np, -1)
    else:
        print('Style undefined.')
        
    if do_plot:
        plt.figure()
        plt.imshow(h_np[:,:,0].T, origin='lower')
        plt.title('initial field')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.show()
    return h_np

############################################ HELPERS

def evaluate_stencil(stencil, fields, origin, domain, title='', do_plot=True):
    
    stencil(
        *fields,
        origin=origin,
        domain=domain)
    
    if do_plot: # careful: plots the last field given.
        plt.figure()
        plt.imshow(np.asarray(fields[-1][:,:,0]).T, origin='lower')
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.show()
        print('Output shape: ', fields[-1].shape)
        
'''        
def run_scheme():
    
    # get staggered variables
    evaluate_stencil(stencil_staggered_x, args_x, origin_staggered, shape_staggered_x, title='staggered x')
    evaluate_stencil(stencil_staggered_y, args_y, origin_staggered, shape_staggered_y, title='staggered y')
    
    # compute default shaped variables baed on the stencil quantities, e.g. derivates, means
    # terms like huMidx[:-1,:] / hMidx[:-1,:]
    # output on default domain
    evaluate_stencil(computations_staggered_x,..., domain=shape_staggered_x,...)
    evaluate_stencil(computations_staggered_y,..., domain=shape_staggered_y,...)
    
    # compute output variables
    # input and output on default domain
    evaluate_stencil(..., domain=shape)
'''
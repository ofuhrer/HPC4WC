# ******************************************************
#     Program: stencil2d
#      Author: Oliver Fuhrer
#       Email: oliverf@meteoswiss.ch
#        Date: 23.06.2022
# Description: Simple stencil example
# ******************************************************

import numpy as np
from mpi4py import MPI

def laplacian(in_field, lap_field, num_halo, extend=0):
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


def update_halo(field, num_halo, p=None):
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
            

def apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=1, p=None):
    """Integrate 4th-order diffusion equation by a certain number of iterations.
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    alpha     -- diffusion coefficient (dimensionless)
    
    Keyword arguments:
    num_iter  -- number of iterations to execute
    """

    tmp_field = np.empty_like(in_field)
    
    for n in range(num_iter):
        
        update_halo(in_field, num_halo, p)
        
        laplacian(in_field, tmp_field, num_halo=num_halo, extend=1)
        laplacian(tmp_field, out_field, num_halo=num_halo, extend=0)
        
        out_field[:, num_halo:-num_halo, num_halo:-num_halo] = \
            in_field[:, num_halo:-num_halo, num_halo:-num_halo] \
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]

        if n < num_iter - 1:
            in_field, out_field = out_field, in_field
        else:
            update_halo(out_field, num_halo, p)
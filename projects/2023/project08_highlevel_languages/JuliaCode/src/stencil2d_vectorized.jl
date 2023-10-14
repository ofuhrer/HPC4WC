module Stencil2d_vectorized

export stencil2d_vectorized

""" Compute the Laplacian using 2nd-order centered differences.

Parameters
----------
in_field : array-like
    Input field (nx x ny x nz with halo in x- and y-direction).
lap_field : array-like
    Result (must be same size as ``in_field``).
num_halo : int
    Number of halo points.
extend : `int`, optional
    Extend computation into halo-zone by this number of points.
"""
function laplacian(in_field, lap_field, num_halo, extend)
    # Preparation
    nx, ny, nz = size(in_field)
    ib = num_halo + 1  - extend
    ie = nx - num_halo + extend
    jb = num_halo + 1  - extend
    je = ny - num_halo + extend

    # Effective computation
    @inbounds @views lap_field[ib:ie, jb:je, :] .= @. ( -4.0 * in_field[ib:ie, jb:je, :]
        + in_field[(ib:ie).-1, jb:je, :]
        + in_field[ib:ie, (jb:je).-1, :]
        + in_field[(ib:ie).+1, jb:je, :]
        + in_field[ib:ie, (jb:je).+1, :])
    return nothing
end


""" Update the halo-zone using an up/down and left/right strategy.

Parameters
----------
field : array-like
    Input/output field (nz x ny x nx with halo in x- and y-direction).
num_halo : int
    Number of halo points.

Note
----
    Corners are updated in the left/right phase of the halo-update.
"""
function update_halo(field, num_halo)
    nx, ny, nz = size(field)

    # left column without corners
    @inbounds @views field[num_halo+1:nx-num_halo, 1:num_halo, :] .= field[num_halo+1:nx-num_halo, ny-2*num_halo+1 : ny-num_halo, :]

    # right column without corners
    @inbounds @views field[num_halo+1:nx-num_halo, ny-num_halo+1:ny, :] .= field[num_halo+1:nx-num_halo, num_halo+1 : 2*num_halo, :]

    # top row with corners
    @inbounds @views field[1:num_halo, :, :] .= field[nx-2*num_halo+1 : nx-num_halo, :, :]

    # bottom row with corners
    @inbounds @views field[nx-num_halo+1:nx, :, :] .= field[num_halo+1 : 2*num_halo, :, :]
    return nothing
end


""" Integrate 4th-order diffusion equation by a certain number of iterations.

Parameters
----------
in_field : array-like
    Input field (nx x ny x nz with halo in x- and y-direction).
lap_field : array-like
    Result (must be same size as ``in_field``).
alpha : float
    Diffusion coefficient (dimensionless).
num_iter : `int`, optional
    Number of iterations to execute.
"""
function apply_diffusion(in_field, out_field, alpha, num_iter, num_halo)
    tmp_field = similar(in_field)
    nx, ny, nz = size(in_field)

    # Iterate over time steps
    for n in 1:num_iter
        @inline update_halo(in_field, num_halo)

        @inline laplacian(in_field, tmp_field, num_halo, 1)
        @inline laplacian(tmp_field, out_field, num_halo, 0)

        @inbounds @views out_field[num_halo+1 : nx-num_halo, num_halo+1 : ny-num_halo, :] .= @. in_field[num_halo+1 : nx-num_halo, num_halo+1 : ny-num_halo, :] - alpha * out_field[num_halo+1 : nx-num_halo, num_halo+1 : ny-num_halo, :]

        if n != num_iter 
            in_field, out_field = out_field, in_field
        else
            @inline update_halo(out_field, num_halo)
        end
    end
    return nothing
end

end #module
module Stencil2d_naive

# Functions that are used elsewhere need to be exported!
export stencil2d_naive, update_halo_naive, laplacian_naive

""" Compute the Laplacian using 2nd-order centered differences.

Parameters
----------
in_field : array-like
    Input field (nz x ny x nx with halo in x- and y-direction).
lap_field : array-like
    Result (must be same size as ``in_field``).
num_halo : int
    Number of halo points.
extend : `int`, optional
    Extend computation into halo-zone by this number of points.
"""

function laplacian(field, lap, num_halo, extend)
    nx = size(field)[1] - 2 * num_halo
    ny = size(field)[2] - 2 * num_halo
    nz = size(field)[3] 

    for k in 1:nz
        for j in 1 + num_halo - extend:ny + num_halo + extend
            for i in 1 + num_halo - extend:nx + num_halo + extend
                lap[i, j, k] = -4.0 * field[i, j, k] +
                field[i - 1, j, k] + field[i + 1, j, k] +
                field[i, j - 1, k] + field[i, j + 1, k]
            end
        end
    end
    
end

""" Integrate 4th-order diffusion equation by a certain number of iterations.

Parameters
----------
in_field : array-like
    Input field (nz x ny x nx with halo in x- and y-direction).
lap_field : array-like
    Result (must be same size as ``in_field``).
alpha : float
    Diffusion coefficient (dimensionless).
num_iter : `int`, optional
    Number of iterations to execute.
"""
function apply_diffusion(in_field, out_field, alpha, num_iter, num_halo)
    nx = size(in_field)[1] - 2 * num_halo
    ny = size(in_field)[2] - 2 * num_halo
    nz = size(in_field)[3] 
    
    tmp1_field = similar(in_field)
    tmp2_field = similar(in_field)
    
    for iter in 1 : num_iter

        update_halo(in_field, num_halo)
        
        laplacian(in_field, tmp1_field, num_halo, 1)
        laplacian(tmp1_field, tmp2_field, num_halo, 0)

        for k in 1:nz
            for j in 1 + num_halo:ny + num_halo
                for i in 1 + num_halo:nx + num_halo
                    out_field[i, j, k] = in_field[i, j, k] - alpha * tmp2_field[i, j, k]
                end
            end
        end
        if iter != num_iter
            for k in 1:nz
                for j in 1 + num_halo:ny + num_halo
                    for i in 1 + num_halo:nx + num_halo
                        in_field[i, j, k] = out_field[i, j, k]
                    end
                end
            end
        end
    
    update_halo(out_field,num_halo)
    end
    return nothing
end

"""
Updates the halo values
"""
function update_halo(field, num_halo)

    nx = size(field)[1] - 2 * num_halo
    ny = size(field)[2] - 2 * num_halo
    nz = size(field)[3] 

    # Bottom edge (without corners)
    for k in 1:nz
        for j in 1:num_halo
            for i in 1 + num_halo:nx + num_halo
                field[i, j, k] = field[i, j + ny, k]
            end
        end
    end

    # Top edge (without corners)
    for k in 1:nz
        for j in ny + num_halo + 1:ny + 2 * num_halo
            for i in 1 + num_halo:nx + num_halo
                field[i, j, k] = field[i, j - ny, k]
            end
        end
    end

    # Left edge (including corners)
    for k in 1:nz
        for j in 1:ny + 2 * num_halo
            for i in 1:num_halo
                field[i, j, k] = field[i + nx, j, k]
            end
        end
    end

    # Right edge (including corners)
    for k in 1:nz
        for j in 1:ny + 2 * num_halo
            for i in nx + num_halo + 1:nx + 2 * num_halo
                field[i, j, k] = field[i - nx, j, k]
            end
        end
    end

    return field
end
        
end #module
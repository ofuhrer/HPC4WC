# ******************************************************
#      Script: Numba Loop
#      Author: HPC4WC Group 7
#        Date: 02.07.2020
# ******************************************************

import numpy as np
from numba import cuda

#additional test functions
@cuda.jit
def increment_by_one(an_array,another_array):
    pos = cuda.grid(1)
    if pos < an_array.size:
        another_array[pos] = an_array[pos]



@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1
        

@cuda.jit
def increment_a_3D_array(an_array):
    x, y, z = cuda.grid(3)
    if x < an_array.shape[0] and y < an_array.shape[1] and z < an_array.shape[2]:
       an_array[x, y, z] += 1
        
        
@cuda.jit
def test(in_field,out_field):
    """
    Simple test function that returns a copy of the in_field.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    
    Returns
    -------
    out_field : a copy of the in_field.
    
    """
    x, y, z = cuda.grid(3)
    if x < in_field.shape[0] and y < in_field.shape[1] and z < in_field.shape[2]:
       out_field[x,y,z] = in_field[x,y,z] 


@cuda.jit
def laplacian1d(in_field, out_field, num_halo=1):
    """Compute Laplacian using 2nd-order centered differences with an explicit nested loop in numba.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : in_field with Laplacian computed in i-direction.
    
    """
    
    i, j, k = cuda.grid(3)
    if i>=num_halo and j>=num_halo and k>=num_halo and i < in_field.shape[0]-num_halo and j < in_field.shape[1]-num_halo and k < in_field.shape[2]-num_halo:
        out_field[i, j, k] = (
            -2.0 * in_field[i, j, k] + in_field[i-1, j, k] + in_field[i+1, j, k])

@cuda.jit
def laplacian2d(in_field, out_field, num_halo=1):
    """
    Compute Laplacian using 2nd-order centered differences with an explicit nested loop in numba.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : in_field with Laplacian computed in i- and j-direction (horizontal Laplacian).
    
    """
    i, j, k = cuda.grid(3)
    if i>=num_halo and j>=num_halo and k>=num_halo and i < in_field.shape[0]-num_halo and j < in_field.shape[1]-num_halo and k < in_field.shape[2]-num_halo:
        out_field[i, j, k] = (
            -4.0 * in_field[i, j, k]
            + in_field[i-1, j, k]
            + in_field[i+1, j, k]
            + in_field[i, j-1, k]
            + in_field[i, j+1, k]
        )
        

@cuda.jit
def laplacian3d(in_field, out_field, num_halo=1):
    """
    Compute Laplacian using 2nd-order centered differences with an explicit nested loop in numba.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : in_field with Laplacian computed in i-, j- and k- direction (full Laplacian).
    
    """

    i, j, k = cuda.grid(3)
    if i>=num_halo and j>=num_halo and k>=num_halo and i < in_field.shape[0]-num_halo and j < in_field.shape[1]-num_halo and k < in_field.shape[2]-num_halo:
                out_field[i, j, k] = (
                    -6.0 * in_field[i, j, k]
                    + in_field[i - 1, j, k]
                    + in_field[i + 1, j, k]
                    + in_field[i, j - 1, k]
                    + in_field[i, j + 1, k]
                    + in_field[i, j, k - 1]
                    + in_field[i, j, k + 1]
                )



@cuda.jit
def FMA(in_field, in_field2, in_field3, out_field, num_halo=0):
    """
    Pointwise stencil to test for fused multiply-add with an explicit nested loop in numba.
    
    Parameters
    ----------
    in_field,in_field2, in_field3  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : fused multiply-add applied to in_field.
    
    """

    i, j, k = cuda.grid(3)
    if i>=num_halo and j>=num_halo and k>=num_halo and i < in_field.shape[0]-num_halo and j < in_field.shape[1]-num_halo and k < in_field.shape[2]-num_halo:

                out_field[i, j, k] = (
                    in_field[i, j, k] + in_field2[i, j, k] * in_field3[i, j, k]
                )


#@cuda.jit
def lapoflap1d(in_field, tmp_field, out_field, num_halo, blockspergrid, threadsperblock):
    """
    Compute Laplacian in i-direction using 2nd-order centered differences with an explicit nested loop in numba.     To ensure synchronization with GPU Threads this is implemented by calling the laplacian function twice.
    
    Parameters
    ----------
    in_field   : input field (nx x ny x nz).
    tmp_field  : intermediate result (must be of same size as in_field).
    out_field : result (must be of same size as in_field).
    num_halo   : number of halo points.
    
    Returns
    -------
    out_field  : Laplacian-of-Laplacian of the input field computed in i-direction.
    
    """
    laplacian1d[blockspergrid, threadsperblock](in_field, tmp_field, num_halo-1)
    laplacian1d[blockspergrid, threadsperblock](tmp_field, out_field, num_halo)
    


#@cuda.jit
def lapoflap2d(in_field, tmp_field, out_field, num_halo,blockspergrid, threadsperblock):
    """
    Compute Laplacian of the Laplacian in i and j-direction using 2nd-order centered differences with an explicit     nested loop in numba. To ensure synchronization with GPU Threads this is implemented by calling the laplacian     function twice.
    
    Parameters
    ----------
    in_field   : input field (nx x ny x nz).
    tmp_field  : intermediate result (must be of same size as in_field).
    out_field : result (must be of same size as in_field).
    num_halo   : number of halo points.
    
    Returns
    -------
    out_field  : Laplacian-of-Laplacian of the input field computed in i- and j-direction (horizontally).
    
    """
    
    laplacian2d[blockspergrid, threadsperblock](in_field, tmp_field, num_halo-1)
    laplacian2d[blockspergrid, threadsperblock](tmp_field, out_field, num_halo)
    

    

#@cuda.jit
def lapoflap3d(in_field, tmp_field, out_field, num_halo,blockspergrid, threadsperblock):
    """
    Compute Laplacian of the Laplacian in i,j,k-direction using 2nd-order centered differences with an explicit    nested loop in numba. To ensure synchronization with GPU Threads this is implemented by calling the laplacian     function twice.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    tmp_field  : intermediate result (must be of same size as in_field).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field  : Laplacian-of-Laplacian of the input field computed in i-, j- and k- direction.
    """
    
    laplacian3d[blockspergrid, threadsperblock](in_field, tmp_field, num_halo-1)
    laplacian3d[blockspergrid, threadsperblock](tmp_field, out_field, num_halo)


import numpy as np
from numba import stencil


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
    out_field[...] = in_field


def laplacian1d(in_field, out_field, num_halo=1):
    """
    Compute the Laplacian of the in_field in i-direction based on a numba stencil kernel for the 2nd-order centered differences.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : in_field with Laplacian computed in i-direction.
    
    """
    # closure inlining
    def laplacian1d_kernel(in_field):
        """
        laplacian kernel which is passed to the numba stencil function
        """
        return -2.0 * in_field[0, 0, 0] + in_field[-1, 0, 0] + in_field[+1, 0, 0]

    out_field = stencil(
        laplacian1d_kernel,
        neighborhood=(
            (-num_halo, num_halo),
            (-num_halo, num_halo),
            (-num_halo, num_halo),
        ),
    )(in_field, out=out_field)

    return out_field


def laplacian2d(in_field, out_field, num_halo=1):
    """
    Compute the Laplacian of the in_field in i- and j-direction based on a numba stencil kernel for the 2nd-order centered differences.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : in_field with Laplacian computed in i- and j-direction (horizontal Laplacian).
    
    """
    # closure inlining
    def laplacian2d_kernel(in_field):
        """
        laplacian kernel which is passed to the numba stencil function
        """
        return (
            -4.0 * in_field[0, 0, 0]
            + in_field[-1, 0, 0]
            + in_field[1, 0, 0]
            + in_field[0, -1, 0]
            + in_field[0, +1, 0]
        )

    out_field = stencil(
        laplacian2d_kernel,
        neighborhood=(
            (-num_halo, num_halo),
            (-num_halo, num_halo),
            (-num_halo, num_halo),
        ),
    )(in_field, out=out_field)

    return out_field


def laplacian3d(in_field, out_field, num_halo=1):
    """
    Compute the Laplacian of the in_field in i-, j- and k-direction based on a numba stencil kernel for the 2nd-order centered differences.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : in_field with Laplacian computed in i-, j- and k- direction (full Laplacian).
    
    """
    # closure inlining
    def laplacian3d_kernel(in_field):
        """
        laplacian kernel which is passed to the numba stencil function
        """
        return (
            -6.0 * in_field[0, 0, 0]
            + in_field[-1, 0, 0]
            + in_field[+1, 0, 0]
            + in_field[0, -1, 0]
            + in_field[0, +1, 0]
            + in_field[0, 0, -1]
            + in_field[0, 0, +1]
        )

    out_field = stencil(
        laplacian3d_kernel,
        neighborhood=(
            (-num_halo, num_halo),
            (-num_halo, num_halo),
            (-num_halo, num_halo),
        ),
    )(in_field, out=out_field)

    return out_field


def FMA(in_field, in_field2, in_field3, out_field, num_halo=0):
    """
    Pointwise stencil to test for fused multiply-add based on a numba stencil kernel
    
    Parameters
    ----------
    in_field,in_field2, in_field3  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : fused multiply-add applied to in_field.
    
    """
    # closure inlining
    def FMA_kernel(in_field, in_field2, in_field3):
        """
        kernel which is passed to the numba stencil function
        """
        return in_field[0, 0, 0] + in_field2[0, 0, 0] * in_field3[0, 0, 0]

    out_field = stencil(
        FMA_kernel,
        neighborhood=(
            (-num_halo, num_halo),
            (-num_halo, num_halo),
            (-num_halo, num_halo),
        ),
    )(in_field, in_field2, in_field3, out=out_field)

    return out_field


def lapoflap1d(in_field, tmp_field, out_field, num_halo=2):
    """
    Compute Laplacian of the Laplacian in i-direction using 2nd-order centered differences based on a numba stencil kernel.
    
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
    # closure inlining
    def laplacian1d_kernel(in_field):
        """
        laplacian kernel which is passed to the numba stencil function
        """
        return -2.0 * in_field[0, 0, 0] + in_field[-1, 0, 0] + in_field[+1, 0, 0]

    tmp_field = stencil(
        laplacian1d_kernel,
        neighborhood=(
            (-num_halo + 1, num_halo - 1),
            (-num_halo + 1, num_halo - 1),
            (-num_halo + 1, num_halo - 1),
        ),
    )(in_field, out=tmp_field)
    out_field = stencil(
        laplacian1d_kernel,
        neighborhood=(
            (-num_halo, num_halo),
            (-num_halo, num_halo),
            (-num_halo, num_halo),
        ),
    )(tmp_field, out=out_field)

    return out_field


def lapoflap2d(in_field, tmp_field, out_field, num_halo=2):
    """
    Compute Laplacian of the Laplacian in i- and j-direction using 2nd-order centered differences based on a numba stencil kernel.
    
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
    # closure inlining
    def laplacian2d_kernel(in_field):
        """
        laplacian kernel which is passed to the numba stencil function
        """
        return (
            -4.0 * in_field[0, 0, 0]
            + in_field[-1, 0, 0]
            + in_field[1, 0, 0]
            + in_field[0, -1, 0]
            + in_field[0, +1, 0]
        )

    tmp_field = stencil(
        laplacian2d_kernel,
        neighborhood=(
            (-num_halo + 1, num_halo - 1),
            (-num_halo + 1, num_halo - 1),
            (-num_halo + 1, num_halo - 1),
        ),
    )(in_field, out=tmp_field)
    out_field = stencil(
        laplacian2d_kernel,
        neighborhood=(
            (-num_halo, num_halo),
            (-num_halo, num_halo),
            (-num_halo, num_halo),
        ),
    )(tmp_field, out=out_field)

    return out_field


def lapoflap3d(in_field, tmp_field, out_field, num_halo=2):
    """
    Compute Laplacian of the Laplacian in i-, j- and k-direction using 2nd-order centered differences based on a numba stencil kernel.
    
    Parameters
    ----------
    in_field   : input field (nx x ny x nz).
    tmp_field  : intermediate result (must be of same size as in_field).
    out_field : result (must be of same size as in_field).
    num_halo   : number of halo points.
    
    Returns
    -------
    out_field  : Laplacian-of-Laplacian of the input field computed in i-, j- and k- direction.
    
    """
    # closure inlining
    def laplacian3d_kernel(in_field):
        """
        laplacian kernel which is passed to the numba stencil function
        """
        return (
            -6.0 * in_field[0, 0, 0]
            + in_field[-1, 0, 0]
            + in_field[+1, 0, 0]
            + in_field[0, -1, 0]
            + in_field[0, +1, 0]
            + in_field[0, 0, -1]
            + in_field[0, 0, +1]
        )

    tmp_field = stencil(
        laplacian3d_kernel,
        neighborhood=(
            (-num_halo + 1, num_halo - 1),
            (-num_halo + 1, num_halo - 1),
            (-num_halo + 1, num_halo - 1),
        ),
    )(in_field, out=tmp_field)
    out_field = stencil(
        laplacian3d_kernel,
        neighborhood=(
            (-num_halo, num_halo),
            (-num_halo, num_halo),
            (-num_halo, num_halo),
        ),
    )(tmp_field, out=out_field)

    return out_field

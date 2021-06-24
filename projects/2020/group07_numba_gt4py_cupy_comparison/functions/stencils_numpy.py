import numpy as np


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

    return out_field


def laplacian1d(in_field, out_field, num_halo=1):
    """
    Compute Laplacian in i-direction using 2nd-order centered differences.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : in_field with Laplacian computed in i-direction.
    
    """

    I, J, K = in_field.shape

    ib = num_halo
    ie = I - num_halo
    jb = num_halo
    je = J - num_halo
    kb = num_halo
    ke = K - num_halo

    out_field[ib:ie, jb:je, kb:ke] = (
        -2.0 * in_field[ib:ie, jb:je, kb:ke]
        + in_field[ib - 1 : ie - 1, jb:je, kb:ke]
        + in_field[ib + 1 : ie + 1, jb:je, kb:ke]
    )

    return out_field


def laplacian2d(in_field, out_field, num_halo=1):
    """
    Compute Laplacian in i- and j-direction using 2nd-order centered differences.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : in_field with Laplacian computed in i- and j-direction (horizontal Laplacian).
    
    """

    I, J, K = in_field.shape

    ib = num_halo
    ie = I - num_halo
    jb = num_halo
    je = J - num_halo
    kb = num_halo
    ke = K - num_halo

    out_field[ib:ie, jb:je, kb:ke] = (
        -4.0 * in_field[ib:ie, jb:je, kb:ke]
        + in_field[ib - 1 : ie - 1, jb:je, kb:ke]
        + in_field[ib + 1 : ie + 1, jb:je, kb:ke]
        + in_field[ib:ie, jb - 1 : je - 1, kb:ke]
        + in_field[ib:ie, jb + 1 : je + 1, kb:ke]
    )

    return out_field


def laplacian3d(in_field, out_field, num_halo=1):
    """
    Compute Laplacian in i-, j- and k-direction using 2nd-order centered differences.
    
    Parameters
    ----------
    in_field  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : in_field with Laplacian computed in i-, j- and k- direction (full Laplacian).
    
    """
    I, J, K = in_field.shape

    ib = num_halo
    ie = I - num_halo
    jb = num_halo
    je = J - num_halo
    kb = num_halo
    ke = K - num_halo

    out_field[ib:ie:, jb:je, kb:ke] = (
        -6.0 * in_field[ib:ie, jb:je, kb:ke]
        + in_field[ib - 1 : ie - 1, jb:je, kb:ke]
        + in_field[ib + 1 : ie + 1, jb:je, kb:ke]
        + in_field[ib:ie, jb - 1 : je - 1, kb:ke]
        + in_field[ib:ie, jb + 1 : je + 1, kb:ke]
        + in_field[ib:ie, jb:je, kb - 1 : ke - 1]
        + in_field[ib:ie, jb:je, kb + 1 : ke + 1]
    )

    return out_field


def FMA(in_field, in_field2, in_field3, out_field, num_halo=0):
    """
    Pointwise stencil to test for fused multiply-add 
    
    Parameters
    ----------
    in_field,in_field2, in_field3  : input field (nx x ny x nz).
    out_field : result (must be of same size as in_field).
    num_halo  : number of halo points.
    
    Returns
    -------
    out_field : fused multiply-add applied to in_field.
    
    """

    I, J, K = in_field.shape

    ib = num_halo
    ie = I - num_halo
    jb = num_halo
    je = J - num_halo
    kb = num_halo
    ke = K - num_halo

    out_field[ib:ie:, jb:je, kb:ke] = (
        in_field[ib:ie:, jb:je, kb:ke]
        + in_field2[ib:ie:, jb:je, kb:ke] * in_field3[ib:ie:, jb:je, kb:ke]
    )

    return out_field


def lapoflap1d(in_field, tmp_field, out_field, num_halo=2):
    """
    Compute Laplacian of the Laplacian in i-direction using 2nd-order centered differences.
    
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

    I, J, K = in_field.shape

    ib = num_halo - 1
    ie = I - num_halo + 1
    jb = num_halo - 1
    je = J - num_halo + 1
    kb = num_halo - 1
    ke = K - num_halo + 1

    tmp_field[ib:ie, jb:je, kb:ke] = (
        -2.0 * in_field[ib:ie, jb:je, kb:ke]
        + in_field[ib - 1 : ie - 1, jb:je, kb:ke]
        + in_field[ib + 1 : ie + 1, jb:je, kb:ke]
    )

    ib = num_halo
    ie = I - num_halo
    jb = num_halo
    je = J - num_halo
    kb = num_halo
    ke = K - num_halo

    out_field[ib:ie, jb:je, kb:ke] = (
        -2.0 * tmp_field[ib:ie, jb:je, kb:ke]
        + tmp_field[ib - 1 : ie - 1, jb:je, kb:ke]
        + tmp_field[ib + 1 : ie + 1, jb:je, kb:ke]
    )

    return out_field


def lapoflap2d(in_field, tmp_field, out_field, num_halo=2):
    """
    Compute Laplacian of the Laplacian in i- and j-direction using 2nd-order centered differences.
    
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
    I, J, K = in_field.shape

    ib = num_halo - 1
    ie = I - num_halo + 1
    jb = num_halo - 1
    je = J - num_halo + 1
    kb = num_halo - 1
    ke = K - num_halo + 1

    tmp_field[ib:ie, jb:je, kb:ke] = (
        -4.0 * in_field[ib:ie, jb:je, kb:ke]
        + in_field[ib - 1 : ie - 1, jb:je, kb:ke]
        + in_field[ib + 1 : ie + 1, jb:je, kb:ke]
        + in_field[ib:ie, jb - 1 : je - 1, kb:ke]
        + in_field[ib:ie, jb + 1 : je + 1, kb:ke]
    )

    ib = num_halo
    ie = I - num_halo
    jb = num_halo
    je = J - num_halo
    kb = num_halo
    ke = K - num_halo

    out_field[ib:ie, jb:je, kb:ke] = (
        -4.0 * tmp_field[ib:ie, jb:je, kb:ke]
        + tmp_field[ib - 1 : ie - 1, jb:je, kb:ke]
        + tmp_field[ib + 1 : ie + 1, jb:je, kb:ke]
        + tmp_field[ib:ie, jb - 1 : je - 1, kb:ke]
        + tmp_field[ib:ie, jb + 1 : ie + 1, kb:ke]
    )

    return out_field


def lapoflap3d(in_field, tmp_field, out_field, num_halo=2):
    """
    Compute Laplacian of the Laplacian in i-, j- and k-direction using 2nd-order centered differences.
    
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

    I, J, K = in_field.shape

    ib = num_halo - 1
    ie = I - num_halo + 1
    jb = num_halo - 1
    je = J - num_halo + 1
    kb = num_halo - 1
    ke = K - num_halo + 1

    tmp_field[ib:ie, jb:je, kb:ke] = (
        -6.0 * in_field[ib:ie, jb:je, kb:ke]
        + in_field[ib - 1 : ie - 1, jb:je, kb:ke]
        + in_field[ib + 1 : ie + 1, jb:je, kb:ke]
        + in_field[ib:ie, jb - 1 : je - 1, kb:ke]
        + in_field[ib:ie, jb + 1 : je + 1, kb:ke]
        + in_field[ib:ie, jb:je, kb - 1 : ke - 1]
        + in_field[ib:ie, jb:je, kb + 1 : ke + 1]
    )

    ib = num_halo
    ie = I - num_halo
    jb = num_halo
    je = J - num_halo
    kb = num_halo
    ke = K - num_halo

    out_field[ib:ie, jb:je, kb:ke] = (
        -6.0 * tmp_field[ib:ie, jb:je, kb:ke]
        + tmp_field[ib - 1 : ie - 1, jb:je, kb:ke]
        + tmp_field[ib + 1 : ie + 1, jb:je, kb:ke]
        + tmp_field[ib:ie, jb - 1 : je - 1, kb:ke]
        + tmp_field[ib:ie, jb + 1 : ie + 1, kb:ke]
        + tmp_field[ib:ie, jb:je, kb - 1 : ke - 1]
        + tmp_field[ib:ie, jb:je, kb + 1 : ie + 1]
    )

    return out_field

import numpy as np


def add_halo_points(field, num_halo):
    """
    Add the halo-zone.
    
    Parameters
    ----------
    field    : input/output 1d: nx, 2d: nx * ny, 3d: nx * ny * nz 
    num_halo : number of halo points
    
    """

    """Adds a halo points to an array on each end (call only once before timeloop)"""
    dim = field.ndim
    if dim == 1:
        nan = np.array(num_halo * [np.nan])
        return np.concatenate((nan, field, nan))
    if dim == 2:
        nan = np.empty((field.shape[0], 1))
        nan[:] = np.nan
        for i in range(num_halo):
            field = np.concatenate((nan, field, nan), axis=1)

        nan = np.empty((1, field.shape[1]))
        nan[:] = np.nan
        for i in range(num_halo):
            field = np.concatenate((nan, field, nan), axis=0)

        return field
    if dim == 3:
        nan = np.empty((field.shape[0], field.shape[1], 1))
        nan[:] = np.nan
        for i in range(num_halo):
            field = np.concatenate((nan, field, nan), axis=2)

        nan = np.empty((field.shape[0], 1, field.shape[2]))
        nan[:] = np.nan
        for i in range(num_halo):
            field = np.concatenate((nan, field, nan), axis=1)

        nan = np.empty((1, field.shape[1], field.shape[2]))
        nan[:] = np.nan
        for i in range(num_halo):
            field = np.concatenate((nan, field, nan), axis=0)

        return field


def update_halo(field, num_halo):
    """
    Update the halo-zone.
    
    Parameters
    ----------
    field    : input/output 1d: nx, 2d: nx * ny, 3d: nx * ny * nz 
    num_halo : number of halo points
    
    """
    dim = field.ndim

    if num_halo == 0:
        field[...] = field
    else:
        if dim == 3:
            # bottom edge
            field[num_halo:-num_halo, 0:num_halo, num_halo:-num_halo] = field[
                num_halo:-num_halo, -2 * num_halo : -num_halo, num_halo:-num_halo
            ]

            # top edge
            field[num_halo:-num_halo, -num_halo:, num_halo:-num_halo] = field[
                num_halo:-num_halo, num_halo : 2 * num_halo, num_halo:-num_halo
            ]

            # left edge
            field[:, :, 0:num_halo] = field[:, :, -2 * num_halo : -num_halo]

            # right edge
            field[:, :, -num_halo:] = field[:, :, num_halo : 2 * num_halo]

            # front edge
            field[0:num_halo, :, :] = field[-2 * num_halo : -num_halo, :, :]

            # back edge
            field[-num_halo:, :, :] = field[num_halo : 2 * num_halo, :, :]
        if dim == 2:
            # bottom edge
            field[:, 0:num_halo] = field[:, -2 * num_halo : -num_halo]
            # top edge
            field[:, -num_halo:] = field[:, num_halo : 2 * num_halo]
            # left edge
            field[0:num_halo, :] = field[-2 * num_halo : -num_halo, :]
            # right edge
            field[-num_halo:, :] = field[num_halo : 2 * num_halo, :]

        if dim == 1:
            # left edge
            field[0:num_halo] = field[-2 * num_halo : -num_halo]
            # right edge
            field[-num_halo:] = field[num_halo : 2 * num_halo]
    return field


def remove_halo_points(field, num_halo): #depreceated!
    """
    Removes halo points to an array on each end (call only once after timeloop before save)
    """
    
    if num_halo == 0:
        field[...]=field
    
    else:
        dim = field.ndim
        if dim == 1:
            field[...] = field[num_halo:-num_halo]
    
        if dim == 2:
            field[...] = field[num_halo:-num_halo, num_halo:-num_halo]
    
        if dim == 3:
            field[...] = field[num_halo:-num_halo, num_halo:-num_halo, num_halo:-num_halo]

    return field

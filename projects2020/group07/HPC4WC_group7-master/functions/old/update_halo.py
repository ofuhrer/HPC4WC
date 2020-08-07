def update_halo(dim, field, num_halo ):
    """Update the halo-zone.
    
    field    -- input/output 1d: nx, 2d: nx * ny, 3d: nx * ny * nz 
    num_halo -- number of halo points
    
    """
    if dim == 3:
        # bottom edge
        field[:, 0:num_halo, num_halo:-num_halo] = field[:, -2 * num_halo:-num_halo, num_halo:-num_halo]

        # top edge
        field[:, -num_halo:, num_halo:-num_halo] = field[:, num_halo:2 * num_halo, num_halo:-num_halo]

        # left edge
        field[:, :, 0:num_halo] = field[:, :, -2 * num_halo:-num_halo]

        # right edge
        field[:, :, -num_halo:] = field[:, :, num_halo:2 * num_halo]
        
        # front edge
        field[0:num_halo, :, :] = field [-2*num_halo:-num_halo, :, :]
        
        #back edge
        field[-num_halo:, :, :] = field [num_halo:2*num_halo, :, :]
    if dim == 2:
        # bottom edge
        field[:,0:num_halo] = field[:, -2*num_halo:-num_halo]
        # top edge
        field[:, -num_halo:] = field[:, num_halo:2*num_halo]
        # left edge
        field[0:num_halo, :] = field[-2*num_halo:-num_halo, :]
        # right edge
        field[-num_halo:, :] = field[num_halo:2*num_halo, :]
        
    if dim == 1:
        # left edge
        field[0:num_halo] = field[-2*num_halo:-num_halo]
        # right edge
        field[-num_halo:] = field[num_halo:2*num_halo]
    return field
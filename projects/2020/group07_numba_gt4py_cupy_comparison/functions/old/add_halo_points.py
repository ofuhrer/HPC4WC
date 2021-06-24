def add_halo_points(dim, field, num_halo):
    import numpy as np
    """Adds a halo points to an array on each end (call only once before timeloop)"""
    if dim == 1:
        nan = np.array( num_halo*[np.nan] )    
        return np.concatenate( (nan, field, nan) )
    if dim == 2:
        nan = np.empty((field.shape[0],1))
        nan[:] = np.nan
        for i in range(num_halo):
            field = np.concatenate((nan, field, nan), axis = 1)
        
        nan = np.empty((1, field.shape[1]))
        nan[:] = np.nan
        for i in range(num_halo):
            field = np.concatenate((nan, field, nan), axis = 0)
        
        return field
    if dim == 3:
        nan = np.empty((field.shape[0], field.shape[1], 1))
        nan[:] = np.nan
        for i in range(num_halo):
            field = np.concatenate((nan, field, nan), axis = 2)
            
        nan = np.empty((field.shape[0], 1, field.shape[2]))
        nan[:] = np.nan
        for i in range(num_halo):
            field = np.concatenate((nan, field, nan), axis = 1)
                   
        nan = np.empty((1, field.shape[1], field.shape[2]))
        nan[:] = np.nan
        for i in range(num_halo):
            field = np.concatenate((nan, field, nan), axis = 0)
         
        
        return field
import cupy as cp
import numpy as np
import math

### Utils
boxplot_options = {
    "showfliers": False,
    "patch_artist": True,
    "notch": False,
    "widths": 2.7,
}

# Halo
def update_halo(field, num_halo):
    # Checks
    dim = len(field.shape)
    assert dim in [2,3]

    # 2d case
    if dim == 2:
        # bottom edge (without corners)
        field[:num_halo, num_halo:-num_halo] = field[
            -2 * num_halo : -num_halo, num_halo:-num_halo
        ]
    
        # top edge (without corners)
        field[-num_halo:, num_halo:-num_halo] = field[
            num_halo : 2 * num_halo, num_halo:-num_halo
        ]
    
        # left edge (including corners)
        field[:, :num_halo] = field[:, -2 * num_halo : -num_halo]
    
        # right edge (including corners)
        field[:, -num_halo:] = field[:, num_halo : 2 * num_halo]
    
    # 3d case
    elif dim == 3:
        # bottom edge (without corners)
        field[:, :num_halo, num_halo:-num_halo] = field[
            :, -2 * num_halo : -num_halo, num_halo:-num_halo
        ]
    
        # top edge (without corners)
        field[:, -num_halo:, num_halo:-num_halo] = field[
            :, num_halo : 2 * num_halo, num_halo:-num_halo
        ]
    
        # left edge (including corners)
        field[:, :, :num_halo] = field[:, :, -2 * num_halo : -num_halo]
    
        # right edge (including corners)
        field[:, :, -num_halo:] = field[:, :, num_halo : 2 * num_halo]
        

# Creates an empty initial field with a square of size / 2 at the center of value `value`
def get_initial_field_square(size, n_halo, value = 1.0) -> cp.ndarray:
    # Check parameters
    assert type(size) == tuple
    dim = len(size)
    # assert dim in [2,3]
    assert dim in [2]

    # Init
    h, w = size[-2], size[-1]

    # Add halo
    h += 2*n_halo
    w += 2*n_halo

    # 2d
    if dim == 2:
        field = cp.zeros((h, w), dtype=cp.float32)
        field[ h//4 : 3*h//4,
               w//4 : 3*w//4 ] = value

    # 3d
    elif dim == 3:
        field = cp.zeros((size[0], h, w), dtype=cp.float32)
        field[ :,
               h//4 : 3*h//4,
               w//4 : 3*w//4 ] = value

    return field

# Creates an grid of spacing `spacing`
def get_initial_field_grid(size, n_halo, value = 1.0, spacing=50) -> cp.ndarray:
    # Check parameters
    assert type(size) == tuple
    dim = len(size)
    # assert dim in [2,3]
    assert dim in [2]

    # Init
    h, w = size[-2], size[-1]

    # Add halo
    h += 2*n_halo
    w += 2*n_halo

    # 2d
    if dim == 2:
        field = cp.zeros((h, w), dtype=cp.float32)

        # Horizontal strides
        for i in range(h // spacing + 2):
            if i * spacing >= h:
                break
            idx = spacing // 3 + i * spacing
            field[ idx : idx + spacing // 3, :] = value
            
        # Vertical strides
        for i in range(w // spacing + 2):
            if i * spacing >= w:
                break
            idx = spacing // 3 + i * spacing
            field[ :, idx : idx + spacing // 3] = value

    # 3d
    elif dim == 3:
        field = cp.zeros((size[0], h, w), dtype=cp.float32)
        
        # Horizontal strides
        for i in range(h // spacing + 2):
            if i * spacing >= h:
                break
            idx = spacing // 3 + i * spacing
            field[ :, idx : idx + spacing // 3, :] = value
            
        # Vertical strides
        for i in range(w // spacing + 2):
            if i * spacing >= w:
                break
            idx = spacing // 3 + i * spacing
            field[ :, :, idx : idx + spacing // 3] = value
        
    return field


### Stencils
# Example with a simple Gaussian filter
def step_stencil_2d_example(in_field, out_field, n_halo):
    # Checks
    assert len(in_field.shape) == 2
    assert len(out_field.shape) == 2
    h,w = out_field.shape
    h_in_,w_in_ = in_field.shape
    assert h_in_ == h + 2*n_halo
    assert w_in_ == w + 2*n_halo

    # IMPORTANT always have an expected halo
    assert n_halo == 1

    # Computation
    out_field[:,:] = (
        4.0 * in_field[1:-1, 1:-1]
        + 2.0 * in_field[2:, 1:-1]
        + 2.0 * in_field[:-2, 1:-1]
        + 2.0 * in_field[1:-1, 2:]
        + 2.0 * in_field[1:-1, :-2]
        + in_field[2:, 2:]
        + in_field[2:, :-2]
        + in_field[:-2, 2:]
        + in_field[:-2, :-2]
    ) / 16.0

    # out_field[:,:] = in_field[2:, 1:-1]

# Example with a simple Gaussian filter
def step_stencil_3d_example(in_field, out_field, n_halo):
    # Checks
    assert len(in_field.shape) == 3
    assert len(out_field.shape) == 3
    _, h,w = out_field.shape
    _, h_in_,w_in_ = in_field.shape
    assert h_in_ == h + 2*n_halo
    assert w_in_ == w + 2*n_halo
    # IMPORTANT always have an expected halo
    assert n_halo == 1

    # Computation
    out_field[:,:,:] = (
        4.0 * in_field[:, 1:-1, 1:-1]
        + 2.0 * in_field[:, 2:, 1:-1]
        + 2.0 * in_field[:, :-2, 1:-1]
        + 2.0 * in_field[:, 1:-1, 2:]
        + 2.0 * in_field[:, 1:-1, :-2]
        + in_field[:, 2:, 2:]
        + in_field[:, 2:, :-2]
        + in_field[:, :-2, 2:]
        + in_field[:, :-2, :-2]
    ) / 16.0
    

# Example with a jacobian stencil
def jacobi_stencil_2d(in_field, out_field, n_halo, alpha=0.5, beta=0.125):
    # Checks
    assert len(in_field.shape) == 2
    assert len(out_field.shape) == 2
    h,w = out_field.shape
    h_in_,w_in_ = in_field.shape
    assert h_in_ == h + 2*n_halo
    assert w_in_ == w + 2*n_halo
    # IMPORTANT always have an expected halo
    assert n_halo == 1

    # Computation
    out_field[:,:] = (
        alpha * in_field[1:-1, 1:-1]
        + beta * ( in_field[2:, 1:-1]+  in_field[:-2, 1:-1]
        +  in_field[1:-1, 2:]+  in_field[1:-1, :-2] )
    )

    
    
# Example with a simple 5x5 Gaussian filter
def gaussian_5x5_stencil_2d(in_field, out_field, n_halo):
    # Checks
    assert len(in_field.shape) == 2
    assert len(out_field.shape) == 2
    h,w = out_field.shape
    h_in_,w_in_ = in_field.shape
    assert h_in_ == h + 2*n_halo
    assert w_in_ == w + 2*n_halo
    
    # IMPORTANT always have an expected halo
    assert n_halo == 2

#     # Kernel parameters
#     if kernel_size == 5:
#         pascal = np.array([[1, 4, 6, 4, 1]])
    
#     kernel = np.transpose(pascal) * np.ones([1, kernel_size]) * pascal
#     kernel /= np.sum(kernel)
    
    # Computation
    out_field[:,:] = (
        (
            in_field[0:-4, 0:-4]
            + 4.0 * in_field[0:-4, 1: -3]
            + 6.0 * in_field[0:-4, 2: -2]
            + 4.0 * in_field[0:-4, 3: -1]
            + in_field[0:-4, 4: ]
        )
        + 4.0 * (
            in_field[1:-3, 0:-4]
            + 4.0 * in_field[1:-3, 1: -3]
            + 6.0 * in_field[1:-3, 2: -2]
            + 4.0 * in_field[1:-3, 3: -1]
            + in_field[1:-3, 4: ]
        )
        + 6.0 * (
            in_field[2:-2, 0:-4]
            + 4.0 * in_field[2:-2, 1: -3]
            + 6.0 * in_field[2:-2, 2: -2]
            + 4.0 * in_field[2:-2, 3: -1]
            + in_field[2:-2, 4: ]
        )
        + 4.0 * (
            in_field[3:-1, 0:-4]
            + 4.0 * in_field[3:-1, 1: -3]
            + 6.0 * in_field[3:-1, 2: -2]
            + 4.0 * in_field[3:-1, 3: -1]
            + in_field[3:-1, 4: ]
        )
        + (
            in_field[4:, 0:-4]
            + 4.0 * in_field[4:, 1: -3]
            + 6.0 * in_field[4:, 2: -2]
            + 4.0 * in_field[4:, 3: -1]
            + in_field[4:, 4: ]
        )
    ) / 256.0
    
    
# Compute
def compute_gpu_2d(in_field, stencil, n_stream, n_iter, n_halo, tile_size = None):
    # Init
    out_field = cp.copy(in_field)
    
    # Check in_field
    dim = len(in_field.shape)
    assert dim == 2
    h,w = in_field.shape
    h -= 2*n_halo
    w -= 2*n_halo
    
    # Chech force tile_size
    if tile_size is None:
        assert math.sqrt(n_stream).is_integer()
        tiles_per_side = (int(math.sqrt(n_stream)), int(math.sqrt(n_stream)))
        h_tile = h // tiles_per_side[0]
        w_tile = w // tiles_per_side[1]
    else:
        assert h % tile_size[0] == 0
        assert w % tile_size[1] == 0
        h_tile, w_tile = tile_size
        tiles_per_side = (h // tile_size[0], w // tile_size[1])

        
    # Create streams
    streams = [ cp.cuda.Stream() for _ in range(n_stream) ]

    for iter in range(n_iter):
        # Init
        e = cp.cuda.Event()
        e.record()

        update_halo(in_field, n_halo)

        # Iterate over tiles
        for i in range(tiles_per_side[0]):
            for j in range(tiles_per_side[1]):
                # # Indeces
                # i, j = (idx // tiles_per_side[0]) % tiles_per_side[0], idx % tiles_per_side[1]
                idx_s = (i*tiles_per_side[0] + j) % n_stream
                with streams[idx_s]:
                    # Stencil iteration
                    # print(f"i = {i}, j = {j}, len in = ({-i*h_stream + 2*n_halo + (i+1)*h_stream}, {-j*w_stream+ 2*n_halo + (j+1)*w_stream}), len out = {(-(n_halo + i*h_stream) + n_halo + (i+1)*h_stream, -( n_halo + j*w_stream) + n_halo + (j+1)*w_stream)}, pos in = [{(i*h_stream , 2*n_halo + (i+1)*h_stream)}, {(j*w_stream , 2*n_halo + (j+1)*w_stream)}], pos out = [{((n_halo + i*h_stream), n_halo + (i+1)*h_stream)}, { ( n_halo + j*w_stream, n_halo + (j+1)*w_stream) }], h_stream = {h_stream}, w_stream  = {w_stream}, h = {h}")
                    stencil(
                        in_field[
                            i*h_tile: 2*n_halo + (i+1)*h_tile,
                            j*w_tile: 2*n_halo + (j+1)*w_tile
                        ],
                        out_field[
                            n_halo + i*h_tile: n_halo + (i+1)*h_tile,
                            n_halo + j*w_tile: n_halo + (j+1)*w_tile
                        ],
                        n_halo
                    )

        # Syncronize all streams
        e.synchronize()

        # Update out_field
        if iter < n_iter - 1:
            in_field, out_field = out_field, in_field
            
    return out_field
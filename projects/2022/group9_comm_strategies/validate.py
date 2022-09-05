## validate stencil output
# TODO: load original outfield, compare new outfield to original one, if the difference is under a certain tolerance it is right

import numpy as np

epsilon = 0.001 # tolerance --> check!

def read_field_from_file(filename, num_halo=None):
    (rank, nbits, num_halo, nx, ny, nz) = np.fromfile(filename, dtype=np.int32, count=6)
    offset=(3 + rank) * 32 // nbits
    data = np.fromfile(filename, dtype=np.float32 if nbits == 32 else np.float64, \
                       count=nz * ny * nx + offset)
    if rank == 3:
        return np.reshape(data[offset:], (nz, ny, nx))
    else:
        return np.reshape(data[offset:], (ny, nx))

in_field_orig = read_field_from_file('in_field_orig.dat')
out_field_orig = read_field_from_file('out_field_orig.dat')

in_field_new = read_field_from_file('in_field.dat')
out_field_new = read_field_from_file('out_field.dat')

if np.allclose(in_field_orig,in_field_new,atol=epsilon) == True:
    print('In-fields validate')
else:
    print('In-fields do not validate')

if np.allclose(out_field_orig,out_field_new,atol=epsilon) == True:
    print('Out-fields validate')
else:
    print('Out-fields do not validate')


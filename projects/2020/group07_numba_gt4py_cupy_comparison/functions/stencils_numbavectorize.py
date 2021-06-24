import numpy as np
from numba import jit, njit, vectorize, stencil, stencils
from numba import vectorize, guvectorize, float64, int32


# vectorize only works for point-wise stencils. maybe @guvectorize does the job for non-point-wise stencils
# did not yet found a way to deal with the halo
@vectorize([float64(float64, float64, float64)], nopython=True)  # target="parallel"
def FMA(in_field, in_field2, in_field3):
    return in_field + in_field2 * in_field3


# @vectorize([float64(float64)])
# def laplacian1d_numbavectorize(in_field):
#    return - 2. * in_field[0, 0, 0]  \
#        + in_field[- 1, 0, 0] + in_field[+ 1 , 0, 0]

# @guvectorize([(float64[:], float64[:], float64, float64)], '(n),(),()->(n)')
# def laplacian1d_numbaguvectorize(in_field, tmp_field, num_halo=1, extend=0 ):
#    for i in range(num_halo -extend, in_field.shape[0] - num_halo + extend):
#        tmp_field[i, : , : ] = - 2. * in_field[i, 0, 0]  \
#        + in_field[i - 1, 0, 0] + in_field[i + 1 , 0, 0]

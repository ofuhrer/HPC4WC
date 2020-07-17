# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: infer_types=True

cimport cython
import numpy as np

def stencil_loop( float[:,:,:] A ):

    cdef int i = 0
    cdef int j = 0
    cdef int k = 0

    cdef int nhalo = 2
    cdef int nx = 37
    cdef int ny = 72
    cdef int nz = 144

    cdef double[:,:,:] C = np.empty((nx,ny,nz), dtype=float)

    with nogil:
        for i in range(nhalo,nx-nhalo):
            for j in range(nhalo,ny-nhalo):
                for k in range(nhalo,nz-nhalo):
                    C[i,j,k] = A[i-1,j,k] + A[i+1,j,k] + A[i,j-1,k] + A[i,j+1,k] + \
                               A[i-1,j-1,k] + A[i+1,j+1,k] + A[i-1,j+1,k] + A[i+1,j-1,k] + \
                               A[i,j,k] + \
                               A[i-1,j,k+1] + A[i+1,j,k+1] + A[i,j-1,k+1] + A[i,j+1,k+1] + \
                               A[i-1,j-1,k+1] + A[i+1,j+1,k+1] + A[i-1,j+1,k+1] + A[i+1,j-1,k+1] + \
                               A[i,j,k+1] + \
                               A[i-1,j,k-1] + A[i+1,j,k-1] + A[i,j-1,k-1] + A[i,j+1,k-1] + \
                               A[i-1,j-1,k-1] + A[i+1,j+1,k-1] + A[i-1,j+1,k-1] + A[i+1,j-1,k-1] + \
                               A[i,j,k-1]

    return np.array(C)

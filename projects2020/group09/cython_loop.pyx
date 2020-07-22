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


def stencil_loop_blocking( float[:,:,:] A):

    cdef int i = 0
    cdef int j = 0
    cdef int k = 0

    cdef int nhalo = 2 # size of array
    cdef int nx = 37
    cdef int ny = 72
    cdef int nz = 144

    cdef int blockx = 3 # regular block sizes
    cdef int blocky = 50
    cdef int blockz = 50

    cdef int blocki = nx//blockx+1 # number of blocks in each direction
    cdef int blockj = ny//blocky+1
    cdef int blockk = nz//blockz+1

    cdef int iblock = 0
    cdef int jblock = 0
    cdef int kblock = 0

    cdef int iblocklen = 0
    cdef int jblocklen = 0
    cdef int kblocklen = 0

    #cdef list blocksizes_x
    #cdef list blocksizes_y
    #cdef list blocksizes_z

    cdef double[:,:,:] C = np.empty((nx,ny,nz), dtype=float)

    with nogil:

        for iblock in range(blocki):
            if iblock == blocki-1: # we are in the last block
                iblocklen = nx%blockx
            else: # we are not in the last block
                iblocklen = blockx
            for jblock in range(blockj):
                if jblock == blockj-1: # we are in the last block
                    jblocklen = ny%blocky
                else: # we are not in the last block
                    jblocklen = blocky
                for kblock in range(blockk):
                    if kblock == blockk-1: # we are in the last block
                        kblocklen = nz%blockz
                    else: # we are not in the last block
                        kblocklen = blockz
                    for i_local in range(iblocklen):
                        for j_local in range(jblocklen):
                            for k_local in range(kblocklen):
                                i = iblock*blockx + i_local
                                j = jblock*blocky + j_local
                                k = kblock*blockz + k_local
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

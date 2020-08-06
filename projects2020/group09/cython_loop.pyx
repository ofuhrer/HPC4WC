# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: infer_types=True
# cython: language_level=3

cimport cython
import numpy as np

def stencil_loop( float[:,:,:] A ):

    cdef int i = 0
    cdef int j = 0
    cdef int k = 0

    cdef int nhalo = 2
    cdef int nx = 3653
    cdef int ny = 720
    cdef int nz = 1440

    cdef double[:,:,:] C = np.empty((nx,ny,nz), dtype=float)

    with nogil:
        for i in range(nhalo,nx-nhalo):
            for j in range(nhalo,ny-nhalo):
                for k in range(nhalo,nz-nhalo):
                    if C[i,j,k] != C[i,j,k]:
                        C[i,j,k] = A[i-2,j,k] + A[i-1,j,k] + A[i,j,k] + A[i+1,j,k] + A[i+2,j,k] + \
                                   A[i-2,j+1,k] + A[i-1,j+1,k] + A[i,j+1,k] + A[i+1,j+1,k] + A[i+2,j+1,k] + \
                                   A[i-2,j+2,k] + A[i-1,j+2,k] + A[i,j+2,k] + A[i+1,j+2,k] + A[i+2,j+2,k] + \
                                   A[i-2,j-1,k] + A[i-1,j-1,k] + A[i,j-1,k] + A[i+1,j-1,k] + A[i+2,j-1,k] + \
                                   A[i-2,j-2,k] + A[i-1,j-2,k] + A[i,j-2,k] + A[i+1,j-2,k] + A[i+2,j-2,k] + \
                                   A[i-2,j,k+1] + A[i-1,j,k+1] + A[i,j,k+1] + A[i+1,j,k+1] + A[i+2,j,k+1] + \
                                   A[i-2,j+1,k+1] + A[i-1,j+1,k+1] + A[i,j+1,k+1] + A[i+1,j+1,k+1] + A[i+2,j+1,k+1] + \
                                   A[i-2,j+2,k+1] + A[i-1,j+2,k+1] + A[i,j+2,k+1] + A[i+1,j+2,k+1] + A[i+2,j+2,k+1] + \
                                   A[i-2,j-1,k+1] + A[i-1,j-1,k+1] + A[i,j-1,k+1] + A[i+1,j-1,k+1] + A[i+2,j-1,k+1] + \
                                   A[i-2,j-2,k+1] + A[i-1,j-2,k+1] + A[i,j-2,k+1] + A[i+1,j-2,k+1] + A[i+2,j-2,k+1] + \
                                   A[i-2,j,k+2] + A[i-1,j,k+2] + A[i,j,k+2] + A[i+1,j,k+2] + A[i+2,j,k+2] + \
                                   A[i-2,j+1,k+2] + A[i-1,j+1,k+2] + A[i,j+1,k+2] + A[i+1,j+1,k+2] + A[i+2,j+1,k+2] + \
                                   A[i-2,j+2,k+2] + A[i-1,j+2,k+2] + A[i,j+2,k+2] + A[i+1,j+2,k+2] + A[i+2,j+2,k+2] + \
                                   A[i-2,j-1,k+2] + A[i-1,j-1,k+2] + A[i,j-1,k+2] + A[i+1,j-1,k+2] + A[i+2,j-1,k+2] + \
                                   A[i-2,j-2,k+2] + A[i-1,j-2,k+2] + A[i,j-2,k+2] + A[i+1,j-2,k+2] + A[i+2,j-2,k+2] + \
                                   A[i-2,j,k-1] + A[i-1,j,k-1] + A[i,j,k-1] + A[i+1,j,k-1] + A[i+2,j,k-1] + \
                                   A[i-2,j+1,k-1] + A[i-1,j+1,k-1] + A[i,j+1,k-1] + A[i+1,j+1,k-1] + A[i+2,j+1,k-1] + \
                                   A[i-2,j+2,k-1] + A[i-1,j+2,k-1] + A[i,j+2,k-1] + A[i+1,j+2,k-1] + A[i+2,j+2,k-1] + \
                                   A[i-2,j-1,k-1] + A[i-1,j-1,k-1] + A[i,j-1,k-1] + A[i+1,j-1,k-1] + A[i+2,j-1,k-1] + \
                                   A[i-2,j-2,k-1] + A[i-1,j-2,k-1] + A[i,j-2,k-1] + A[i+1,j-2,k-1] + A[i+2,j-2,k-1] + \
                                   A[i-2,j,k-2] + A[i-1,j,k-2] + A[i,j,k-2] + A[i+1,j,k-2] + A[i+2,j,k-2] + \
                                   A[i-2,j+1,k-2] + A[i-1,j+1,k-2] + A[i,j+1,k-2] + A[i+1,j+1,k-2] + A[i+2,j+1,k-2] + \
                                   A[i-2,j+2,k-2] + A[i-1,j+2,k-2] + A[i,j+2,k-2] + A[i+1,j+2,k-2] + A[i+2,j+2,k-2] + \
                                   A[i-2,j-1,k-2] + A[i-1,j-1,k-2] + A[i,j-1,k-2] + A[i+1,j-1,k-2] + A[i+2,j-1,k-2] + \
                                   A[i-2,j-2,k-2] + A[i-1,j-2,k-2] + A[i,j-2,k-2] + A[i+1,j-2,k-2] + A[i+2,j-2,k-2]
                    else:
                        C[i,j,k] = A[i,j,k]

    return np.array(C)


def stencil_loop_blocking( float[:,:,:] A):

    cdef int i = 0
    cdef int j = 0
    cdef int k = 0

    cdef int nhalo = 2 # size of array
    cdef int nx = 3653
    cdef int ny = 720
    cdef int nz = 1440

    cdef int blockx = 44 # regular block sizes
    cdef int blocky = 44
    cdef int blockz = 44

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
                iblocklen = nx%blockx-nhalo
            else: # we are not in the last block
                iblocklen = blockx
            for jblock in range(blockj):
                if jblock == blockj-1: # we are in the last block
                    jblocklen = ny%blocky-nhalo
                else: # we are not in the last block
                    jblocklen = blocky
                for kblock in range(blockk):
                    if kblock == blockk-1: # we are in the last block
                        kblocklen = nz%blockz-nhalo
                    else: # we are not in the last block
                        kblocklen = blockz
                    for i_local in range(iblocklen):
                        for j_local in range(jblocklen):
                            for k_local in range(kblocklen):
                                i = iblock*blockx + i_local
                                j = jblock*blocky + j_local
                                k = kblock*blockz + k_local
                                if (i >= nhalo) & (j >= nhalo) & (k >= nhalo):
                                    if C[i,j,k] != C[i,j,k]:
                                        C[i,j,k] = A[i-2,j,k] + A[i-1,j,k] + A[i,j,k] + A[i+1,j,k] + A[i+2,j,k] + \
                                                   A[i-2,j+1,k] + A[i-1,j+1,k] + A[i,j+1,k] + A[i+1,j+1,k] + A[i+2,j+1,k] + \
                                                   A[i-2,j+2,k] + A[i-1,j+2,k] + A[i,j+2,k] + A[i+1,j+2,k] + A[i+2,j+2,k] + \
                                                   A[i-2,j-1,k] + A[i-1,j-1,k] + A[i,j-1,k] + A[i+1,j-1,k] + A[i+2,j-1,k] + \
                                                   A[i-2,j-2,k] + A[i-1,j-2,k] + A[i,j-2,k] + A[i+1,j-2,k] + A[i+2,j-2,k] + \
                                                   A[i-2,j,k+1] + A[i-1,j,k+1] + A[i,j,k+1] + A[i+1,j,k+1] + A[i+2,j,k+1] + \
                                                   A[i-2,j+1,k+1] + A[i-1,j+1,k+1] + A[i,j+1,k+1] + A[i+1,j+1,k+1] + A[i+2,j+1,k+1] + \
                                                   A[i-2,j+2,k+1] + A[i-1,j+2,k+1] + A[i,j+2,k+1] + A[i+1,j+2,k+1] + A[i+2,j+2,k+1] + \
                                                   A[i-2,j-1,k+1] + A[i-1,j-1,k+1] + A[i,j-1,k+1] + A[i+1,j-1,k+1] + A[i+2,j-1,k+1] + \
                                                   A[i-2,j-2,k+1] + A[i-1,j-2,k+1] + A[i,j-2,k+1] + A[i+1,j-2,k+1] + A[i+2,j-2,k+1] + \
                                                   A[i-2,j,k+2] + A[i-1,j,k+2] + A[i,j,k+2] + A[i+1,j,k+2] + A[i+2,j,k+2] + \
                                                   A[i-2,j+1,k+2] + A[i-1,j+1,k+2] + A[i,j+1,k+2] + A[i+1,j+1,k+2] + A[i+2,j+1,k+2] + \
                                                   A[i-2,j+2,k+2] + A[i-1,j+2,k+2] + A[i,j+2,k+2] + A[i+1,j+2,k+2] + A[i+2,j+2,k+2] + \
                                                   A[i-2,j-1,k+2] + A[i-1,j-1,k+2] + A[i,j-1,k+2] + A[i+1,j-1,k+2] + A[i+2,j-1,k+2] + \
                                                   A[i-2,j-2,k+2] + A[i-1,j-2,k+2] + A[i,j-2,k+2] + A[i+1,j-2,k+2] + A[i+2,j-2,k+2] + \
                                                   A[i-2,j,k-1] + A[i-1,j,k-1] + A[i,j,k-1] + A[i+1,j,k-1] + A[i+2,j,k-1] + \
                                                   A[i-2,j+1,k-1] + A[i-1,j+1,k-1] + A[i,j+1,k-1] + A[i+1,j+1,k-1] + A[i+2,j+1,k-1] + \
                                                   A[i-2,j+2,k-1] + A[i-1,j+2,k-1] + A[i,j+2,k-1] + A[i+1,j+2,k-1] + A[i+2,j+2,k-1] + \
                                                   A[i-2,j-1,k-1] + A[i-1,j-1,k-1] + A[i,j-1,k-1] + A[i+1,j-1,k-1] + A[i+2,j-1,k-1] + \
                                                   A[i-2,j-2,k-1] + A[i-1,j-2,k-1] + A[i,j-2,k-1] + A[i+1,j-2,k-1] + A[i+2,j-2,k-1] + \
                                                   A[i-2,j,k-2] + A[i-1,j,k-2] + A[i,j,k-2] + A[i+1,j,k-2] + A[i+2,j,k-2] + \
                                                   A[i-2,j+1,k-2] + A[i-1,j+1,k-2] + A[i,j+1,k-2] + A[i+1,j+1,k-2] + A[i+2,j+1,k-2] + \
                                                   A[i-2,j+2,k-2] + A[i-1,j+2,k-2] + A[i,j+2,k-2] + A[i+1,j+2,k-2] + A[i+2,j+2,k-2] + \
                                                   A[i-2,j-1,k-2] + A[i-1,j-1,k-2] + A[i,j-1,k-2] + A[i+1,j-1,k-2] + A[i+2,j-1,k-2] + \
                                                   A[i-2,j-2,k-2] + A[i-1,j-2,k-2] + A[i,j-2,k-2] + A[i+1,j-2,k-2] + A[i+2,j-2,k-2]
                                    else:
                                        C[i,j,k] = A[i,j,k]
    return np.array(C)

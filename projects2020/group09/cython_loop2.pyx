# cython: language_level=3

import numpy as np
cimport numpy as np
from cython import boundscheck, wraparound, cdivision

ctypedef np.float64_t DTYPE_t

@boundscheck(False)
@cdivision(True)
@wraparound(False)
def run(np.ndarray[DTYPE_t, ndim=4] arr):
    cdef int nvar = arr.shape[0]
    cdef int nt = arr.shape[1]
    cdef int nlat = arr.shape[2]
    cdef int nlon = arr.shape[3]
    cdef DTYPE_t [:, :, :, :] res = np.empty((nvar, nt, nlat, nlon), dtype=arr.dtype)
    cdef int ivar, it, ilat, ilon
    cdef int jt, jlat, jlon
    cdef int iteff, ilateff, iloneff
    cdef DTYPE_t val, valsum
    cdef int count
    for ivar in range(nvar):
        for it in range(nt):
            for ilat in range(nlat):
                for ilon in range(nlon):
                    val = arr[ivar, it, ilat, ilon]
                    if (val != val):
                        valsum = 0.
                        count = 0
                        for jt in range(-2, 3):
                            iteff = it + jt
                            if iteff < 0 or iteff > nt-1:
                                break
                            for jlat in range(-2, 3):
                                ilateff = ilat + jlat
                                if ilateff < 0 or ilateff > nlat-1:
                                    break
                                for jlon in range(-2, 3):
                                    iloneff = ilon + jlon
                                    if iloneff < 0 or iloneff > nlon-1:
                                        break
                                    val = arr[ivar, iteff, ilateff, iloneff]
                                    if val == val:
                                        valsum += val
                                        count += 1
                        if count > 0:
                            res[ivar, it, ilat, ilon] = valsum / count
                        else:
                            res[ivar, it, ilat, ilon] = val
                    else:
                        res[ivar, it, ilat, ilon] = val
    return res

def run_block(np.ndarray[DTYPE_t, ndim=4] arr):
    cdef int nvar = arr.shape[0]
    cdef int nt = arr.shape[1]
    cdef int nlat = arr.shape[2]
    cdef int nlon = arr.shape[3]
    cdef DTYPE_t [:, :, :, :] res = np.empty((nvar, nt, nlat, nlon), dtype=arr.dtype)
    cdef int ivar, it, ilat, ilon
    cdef int jt, jlat, jlon
    cdef int iteff, ilateff, iloneff
    cdef DTYPE_t val, valsum
    cdef int count

    cdef int nhalo = 2 # size of array
    cdef int nx = 37
    cdef int ny = 720
    cdef int nz = 1440

    cdef int blockx = 10 # regular block sizes
    cdef int blocky = 500
    cdef int blockz = 500

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
    for ivar in range(nvar):
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
                                ilat = jblock*blocky + j_local
                                ilon = kblock*blockz + k_local
                                val = arr[ivar, it,ilat,ilon]
                                if val != val:
                                    valsum = 0.
                                    count = 0
                                    for jt in range(-2, 3):
                                        iteff = it + jt
                                        if iteff < 0 or iteff > nt-1:
                                            break
                                        for jlat in range(-2, 3):
                                            ilateff = ilat + jlat
                                            if ilateff < 0 or ilateff > nlat-1:
                                                break
                                            for jlon in range(-2, 3):
                                                iloneff = ilon + jlon
                                                if iloneff < 0 or iloneff > nlon-1:
                                                    break
                                                val = arr[ivar, iteff, ilateff, iloneff]
                                                if val == val:
                                                    valsum += val
                                                    count += 1
                                    if count > 0:
                                        res[ivar, it, ilat, ilon] = valsum / count
                                    else:
                                        res[ivar, it, ilat, ilon] = val
                                else:
                                    res[ivar, it, ilat, ilon] = val
    return res

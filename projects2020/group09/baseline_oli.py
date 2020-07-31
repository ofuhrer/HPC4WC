"""
Example for workflow for gapfilling remote sensing data from diverse sources

    @author: verena bessenbacher
    @date: 12 06 2020
"""

import numpy as np
import numba
from datetime import datetime
import xarray as xr
from scipy.ndimage.filters import generic_filter

frac_missing = 0.42
filepath = '/net/so4/landclim/bverena/large_files/data_small.nc'

# create example array
print(f'read data array')
data = xr.open_dataarray(filepath)

# subset more for speedup of first tests
#print(f'subset even more because very large dataset')
#data = data[:,::100,:,:]
print(data.shape)

# numpy 
tic = datetime.now()
footprint = np.ones((1,5,5,5))
tmp = generic_filter(data, np.nanmean, footprint=footprint, mode='nearest')
toc = datetime.now()
print(f'numpy {toc-tic}')

# numba njit
@numba.njit
def numba_nanmean(values):
    return np.nanmean(values)
tic = datetime.now()
footprint = np.ones((1,3,3,3))
tmp = generic_filter(data, numba_nanmean, footprint=footprint, mode='nearest')
toc = datetime.now()
print(f'numba {toc-tic}')

# numba stencil
@numba.stencil(neighborhood = ((-2,2),(-2,2),(-2,2)))
def _sum(w):
    return (w[-1,0,0] + w[+1,0,0] + w[0,-1,0] + w[0,+1,0] +
            w[-1,-1,0] + w[+1,+1,0] + w[-1,+1,0] + w[+1,-1,0] + w[0,0,0] +

            w[-1,0,1] + w[+1,0,1] + w[0,-1,1] + w[0,+1,1] +
            w[-1,-1,1] + w[+1,+1,1] + w[-1,+1,1] + w[+1,-1,1] + w[0,0,1] +

            w[-1,0,-1] + w[+1,0,-1] + w[0,-1,-1] + w[0,+1,-1] +
            w[-1,-1,-1] + w[+1,+1,-1] + w[-1,+1,-1] + w[+1,-1,-1] + w[0,0,-1])

mask = ~ np.isnan(data.values)
datanum = np.nan_to_num(data)
tic = datetime.now()
result = np.empty(data.shape)
weights = np.empty(data.shape)
result[0,:,:,:] = _sum(datanum[0,:,:,:])
result[1,:,:,:] = _sum(datanum[1,:,:,:])
result[2,:,:,:] = _sum(datanum[2,:,:,:])
weights[0,:,:,:] = _sum(mask[0,:,:,:])
weights[1,:,:,:] = _sum(mask[1,:,:,:])
weights[2,:,:,:] = _sum(mask[2,:,:,:])
result = result / weights
result = np.where(weights == 0, np.nan, result)
toc = datetime.now()
vali = np.isclose(result[:,2:-2,2:-2,2:-2], tmp[:,2:-2,2:-2,2:-2], equal_nan=True).all()
print(f'stencil {toc-tic}, validated: {vali}')

# numba stencil and njit
@numba.njit
def numba_sum(w):
    return _sum(w)

tic = datetime.now()
result = np.empty(data.shape)
weights = np.empty(data.shape)
result[0,:,:,:] = numba_sum(datanum[0,:,:,:])
result[1,:,:,:] = numba_sum(datanum[1,:,:,:])
result[2,:,:,:] = numba_sum(datanum[2,:,:,:])
weights[0,:,:,:] = numba_sum(mask[0,:,:,:])
weights[1,:,:,:] = numba_sum(mask[1,:,:,:])
weights[2,:,:,:] = numba_sum(mask[2,:,:,:])
result = result / weights
result = np.where(weights == 0, np.nan, result)
toc = datetime.now()
vali = np.isclose(result[:,2:-2,2:-2,2:-2], tmp[:,2:-2,2:-2,2:-2], equal_nan=True).all()
print(f'stencil + njit {toc-tic}, validated: {vali}')

# cython stencil
from cython_loop import stencil_loop
mask = (~ np.isnan(data.values) *1.)
mask = mask.astype(np.float32)
tic = datetime.now()
result = np.empty(data.shape, dtype=np.float32)
weights = np.empty(data.shape, dtype=np.float32)
result[0,:,:,:] = stencil_loop(datanum[0,:,:,:])
result[1,:,:,:] = stencil_loop(datanum[1,:,:,:])
result[2,:,:,:] = stencil_loop(datanum[2,:,:,:])
weights[0,:,:,:] = stencil_loop(mask[0,:,:,:])
weights[1,:,:,:] = stencil_loop(mask[1,:,:,:])
weights[2,:,:,:] = stencil_loop(mask[2,:,:,:])
result = result / weights
result = np.where(weights == 0, np.nan, result)
toc = datetime.now()
vali = np.isclose(result[:,2:-2,2:-2,2:-2], tmp[:,2:-2,2:-2,2:-2], equal_nan=True).all()
print(f'cython {toc-tic}, validated: {vali}')

# cython stencil
from cython_loop import stencil_loop_blocking as stencil_loop
mask = (~ np.isnan(data.values) *1.)
mask = mask.astype(np.float32)
tic = datetime.now()
result = np.empty(data.shape, dtype=np.float32)
weights = np.empty(data.shape, dtype=np.float32)
result[0,:,:,:] = stencil_loop(datanum[0,:,:,:])
result[1,:,:,:] = stencil_loop(datanum[1,:,:,:])
result[2,:,:,:] = stencil_loop(datanum[2,:,:,:])
weights[0,:,:,:] = stencil_loop(mask[0,:,:,:])
weights[1,:,:,:] = stencil_loop(mask[1,:,:,:])
weights[2,:,:,:] = stencil_loop(mask[2,:,:,:])
result = result / weights
result = np.where(weights == 0, np.nan, result)
toc = datetime.now()
vali = np.isclose(result[:,2:-2,2:-2,2:-2], tmp[:,2:-2,2:-2,2:-2], equal_nan=True).all()
print(f'cython block {toc-tic}, validated: {vali}')

# stencil debugging area
nx = 11 # total size
ny = 11
nz = 11
blockx = 5 # block size
blocky = 5
blockz = 5
blocki = nx//blockx+1 # number of blocks (last one smaller)
blockj = ny//blocky+1
blockk = nz//blockz+1

blocksizes_x = [blockx] * (nx//blockx) + [nx%blockx]
blocksizes_y = [blocky] * (ny//blocky) + [ny%blocky]
blocksizes_z = [blockz] * (nz//blockz) + [nz%blockz]

A = np.zeros((nx,ny,nz))
for iblock, iblocklen in enumerate(blocksizes_x):
    for jblock, jblocklen in enumerate(blocksizes_y):
        for kblock, kblocklen in enumerate(blocksizes_z):
            for i_local in range(iblocklen):
                for j_local in range(jblocklen):
                    for k_local in range(kblocklen):
                        i = iblock*blockx + i_local
                        j = jblock*blocky + j_local
                        k = kblock*blockz + k_local
                        A[i,j,k] = 1

A = np.zeros((nx,ny,nz))
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
        #print(jblock, blockj, jblocklen)
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
                        A[i,j,k] = 1
#data = data.fillna(tmp)
# save result as "ground truth" for testing other approaches
#data.to_netcdf('baseline_result.nc')
# my PhD Project goes on with:
# gapfill each variable by regressing over all the others
# in an iterative EM-like fashion 
# with spatiotemporal gapfill as initial guess
# until estimates for missing values converge

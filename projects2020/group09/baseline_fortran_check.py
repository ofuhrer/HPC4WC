"""
Test for the fortran script for gapfilling remote sensing data from diverse sources
This test is slow, but working (I only use it for small example fields).

    @author: Ulrike Proske
    @date: 17 07 2020
"""

import numpy as np
import numba
from datetime import datetime
import xarray as xr
from scipy.ndimage.filters import generic_filter

# absolute tolerance for the isclose validation:
abstol = 0.05
filepath = './foin.nc'

# create example array
print(f'read data array')
data = xr.open_dataarray(filepath)
data = data[:,:,:,:]
print(data.shape)
for i in range(0,np.shape(data)[0]):
    print(i)
    for j in range(0,np.shape(data)[1]):
        for k in range(0,np.shape(data)[2]):
            for h in range(0,np.shape(data)[3]):
                if data[i,j,k,h] == 0:
                    data[i,j,k,h] = np.nan

# numpy 
tic = datetime.now()
footprint = np.ones((5,5,5,1))
tmp = generic_filter(data, np.nanmean, footprint=footprint, mode='nearest')
toc = datetime.now()
print(f'numpy {toc-tic}')
result = xr.open_dataarray('./fout.nc')
result = result[:,:,:,:]
# debug import IPython; IPython.embed()
# where tmp is nan, this should be changed back to 0, because that's what the fortran code deals with instead of nans:
for i in range(0,np.shape(tmp)[0]):
    print(i)
    for j in range(0,np.shape(tmp)[1]):
        for k in range(0,np.shape(tmp)[2]):
            for h in range(0,np.shape(tmp)[3]):
                if np.isnan(tmp[i,j,k,h]):
                    tmp[i,j,k,h] = 0

vali = np.isclose(result[2:-2,2:-2,2:-2,:], tmp[2:-2,2:-2,2:-2,:], atol=abstol, equal_nan=True).all()
print(f'validated: {vali}, absolute tolerance: {abstol}')

"""
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
@numba.stencil
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
"""

#data = data.fillna(tmp)
# save result as "ground truth" for testing other approaches
#data.to_netcdf('baseline_result.nc')
# my PhD Project goes on with:
# gapfill each variable by regressing over all the others
# in an iterative EM-like fashion 
# with spatiotemporal gapfill as initial guess
# until estimates for missing values converge

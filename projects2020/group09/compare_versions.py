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
irun = False

# load data
data_flat = xr.open_dataset('/net/so4/landclim/bverena/large_files/data_large.nc')

# ocean points are omitted, reverse
lsm = xr.open_dataarray('/net/so4/landclim/bverena/large_files/landmask_idebug_False.nc')
landlat, landlon = np.where(lsm)
data = np.full((3,3653,720,1440), np.nan)
data[:,:,landlat,landlon] = data_flat['tp']

# make smaller for faster comparison
#data = data[:,::100,:,:]
data = data.astype(np.float32)
data = xr.DataArray(data)
print(f'data shape is {data.shape}')

# numpy 
tic = datetime.now()
footprint = np.ones((1,5,5,5))
#tmp = generic_filter(data, np.nanmean, footprint=footprint, mode='nearest')
toc = datetime.now()
#print(f'numpy {toc-tic}')
print(f'numpy omitted for speed')

# numba njit
if False:
    @numba.njit
    def numba_nanmean(values):
        return np.nanmean(values)
    tic = datetime.now()
    footprint = np.ones((1,5,5,5))
    tmp = generic_filter(data, numba_nanmean, footprint=footprint, mode='nearest')
    toc = datetime.now()
    print(f'numba.njit {toc-tic}')

# numba stencil
    @numba.stencil
    def _sum(w):
        return (w[-2,-2,2] + w[-2,-1,2] + w[-2,0,2] + w[-2,+1,2] + w[-2,+2,2] +
                w[-1,-2,2] + w[-1,-1,2] + w[-1,0,2] + w[-1,+1,2] + w[-1,+2,2] +
                w[0,-2,2] + w[0,-1,2] + w[0,0,2] + w[0,+1,2] + w[0,+2,2] +
                w[+1,-2,2] + w[+1,-1,2] + w[+1,0,2] + w[+1,+1,2] + w[+1,+2,2] +
                w[+2,-2,2] + w[+2,-1,2] + w[+2,0,2] + w[+2,+1,2] + w[+2,+2,2] + 

                w[-2,-2,1] + w[-2,-1,1] + w[-2,0,1] + w[-2,+1,1] + w[-2,+2,1] +
                w[-1,-2,1] + w[-1,-1,1] + w[-1,0,1] + w[-1,+1,1] + w[-1,+2,1] +
                w[0,-2,1] + w[0,-1,1] + w[0,0,1] + w[0,+1,1] + w[0,+2,1] +
                w[+1,-2,1] + w[+1,-1,1] + w[+1,0,1] + w[+1,+1,1] + w[+1,+2,1] +
                w[+2,-2,1] + w[+2,-1,1] + w[+2,0,1] + w[+2,+1,1] + w[+2,+2,1] + 

                w[-2,-2,0] + w[-2,-1,0] + w[-2,0,0] + w[-2,+1,0] + w[-2,+2,0] +
                w[-1,-2,0] + w[-1,-1,0] + w[-1,0,0] + w[-1,+1,0] + w[-1,+2,0] +
                w[0,-2,0] + w[0,-1,0] + w[0,0,0] + w[0,+1,0] + w[0,+2,0] + # w point itself!
                w[+1,-2,0] + w[+1,-1,0] + w[+1,0,0] + w[+1,+1,0] + w[+1,+2,0] +
                w[+2,-2,0] + w[+2,-1,0] + w[+2,0,0] + w[+2,+1,0] + w[+2,+2,0] + 

                w[-2,-2,-1] + w[-2,-1,-1] + w[-2,0,-1] + w[-2,+1,-1] + w[-2,+2,-1] +
                w[-1,-2,-1] + w[-1,-1,-1] + w[-1,0,-1] + w[-1,+1,-1] + w[-1,+2,-1] +
                w[0,-2,-1] + w[0,-1,-1] + w[0,0,-1] + w[0,+1,-1] + w[0,+2,-1] +
                w[+1,-2,-1] + w[+1,-1,-1] + w[+1,0,-1] + w[+1,+1,-1] + w[+1,+2,-1] +
                w[+2,-2,-1] + w[+2,-1,-1] + w[+2,0,-1] + w[+2,+1,-1] + w[+2,+2,-1] + 

                w[-2,-2,-2] + w[-2,-1,-2] + w[-2,0,-2] + w[-2,+1,-2] + w[-2,+2,-2] +
                w[-1,-2,-2] + w[-1,-1,-2] + w[-1,0,-2] + w[-1,+1,-2] + w[-1,+2,-2] +
                w[0,-2,-2] + w[0,-1,-2] + w[0,0,-2] + w[0,+1,-2] + w[0,+2,-2] +
                w[+1,-2,-2] + w[+1,-1,-2] + w[+1,0,-2] + w[+1,+1,-2] + w[+1,+2,-2] +
                w[+2,-2,-2] + w[+2,-1,-2] + w[+2,0,-2] + w[+2,+1,-2] + w[+2,+2,-2])

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
    print(f'numba.stencil {toc-tic}, validated: {vali}')

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
    print(f'numba.stencil + numba.njit {toc-tic}, validated: {vali}')

# cython stencil
from cython_loop import stencil_loop
datanum = np.nan_to_num(data)
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
tmp = result
toc = datetime.now()
vali = np.isclose(result[:,2:-2,2:-2,2:-2], tmp[:,2:-2,2:-2,2:-2], equal_nan=True).all()
print(f'cython with stencil {toc-tic}, validated: {vali}')

# cython stencil oli
from cython_loop2 import run
tic = datetime.now()
data = data.astype(np.float64)
result = run(data.values)
toc = datetime.now()
vali = np.isclose(result[:,2:-2,2:-2,2:-2], tmp[:,2:-2,2:-2,2:-2], equal_nan=True).all()
print(f'cython with loop {toc-tic}, validated: {vali}')

# cython stencil oli blocked
from cython_loop2 import run_block
tic = datetime.now()
data = data.astype(np.float64)
result = run_block(data.values)
toc = datetime.now()
vali = np.isclose(result[:,2:-2,2:-2,2:-2], tmp[:,2:-2,2:-2,2:-2], equal_nan=True).all()
print(f'cython with loop, blocked {toc-tic}, validated: {vali}')

# cython stencil with blocking
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
print(f'cython stencil, blocked {toc-tic}, validated: {vali}')

# parallelise over variables with concurrent.futures
from concurrent.futures.thread import ThreadPoolExecutor
from cython_loop import stencil_loop
datanum = np.nan_to_num(data)
mask = (~ np.isnan(data.values) *1.)
mask = mask.astype(np.float32)
tic = datetime.now()
result = np.empty(data.shape, dtype=np.float32)
weights = np.empty(data.shape, dtype=np.float32)
with ThreadPoolExecutor(max_workers=6) as executor:
    v1 = executor.submit(stencil_loop, datanum[0,:,:,:])
    v2 = executor.submit(stencil_loop, datanum[1,:,:,:])
    v3 = executor.submit(stencil_loop, datanum[2,:,:,:])

    w1 = executor.submit(stencil_loop, mask[0,:,:,:])
    w2 = executor.submit(stencil_loop, mask[1,:,:,:])
    w3 = executor.submit(stencil_loop, mask[2,:,:,:])

    result[0,:,:,:] = v1.result()
    result[1,:,:,:] = v2.result()
    result[2,:,:,:] = v3.result()

    weights[0,:,:,:] = w1.result()
    weights[1,:,:,:] = w2.result()
    weights[2,:,:,:] = w3.result()

result = result / weights
result = np.where(weights == 0, np.nan, result)
toc = datetime.now()
vali = np.isclose(result[:,2:-2,2:-2,2:-2], tmp[:,2:-2,2:-2,2:-2], equal_nan=True).all()
print(f'concurrent futures parAllel over vars with cython stencil {toc-tic}, validated: {vali}')
#print(f'concurrent futures parallel over vars with cython stencil {toc-tic}')#, validated: {vali}')

"""
Example for workflow for gapfilling remote sensing data from diverse sources
Explicit stencil filter

sources:
http://numba.pydata.org/numba-doc/latest/user/stencil.html
https://examples.dask.org/applications/stencils-with-numba.html

    @author: verena bessenbacher
    @date: 12 06 2020
"""

import numpy as np
from datetime import datetime
import xarray as xr
import numba

@numba.stencil
def _nanmean(v, w):
    return (v[-1,0,0] * w[-1,0,0] + v[+1,0,0] * w[+1,0,0] +
            v[0,-1,0] * w[0,-1,0] + v[0,+1,0] * w[0,+1,0] +
            v[-1,-1,0] * w[-1,-1,0] + v[+1,+1,0] * w[+1,+1,0] +
            v[-1,+1,0] * w[-1,+1,0] + v[+1,-1,0] * w[+1,-1,0] +

            v[-1,0,1] * w[-1,0,1] + v[+1,0,1] * w[+1,0,1] +
            v[0,-1,1] * w[0,-1,1] + v[0,+1,1] * w[0,+1,1] +
            v[-1,-1,1] * w[-1,-1,1] + v[+1,+1,1] * w[+1,+1,1] +
            v[-1,+1,1] * w[-1,+1,1] + v[+1,-1,1] * w[+1,-1,1] +
            v[0,0,1] * w[0,0,1] +

            v[-1,0,-1] * w[-1,0,-1] + v[+1,0,-1] * w[+1,0,-1] +
            v[0,-1,-1] * w[0,-1,-1] + v[0,+1,-1] * w[0,+1,-1] +
            v[-1,-1,-1] * w[-1,-1,-1] + v[+1,+1,-1] * w[+1,+1,-1] +
            v[-1,+1,-1] * w[-1,+1,-1] + v[+1,-1,-1] * w[+1,-1,-1] +
            v[0,0,-1] * w[0,0,-1]) / max(w[-1,0,0] + 
            
            w[+1,0,0] + w[0,-1,0] + w[0,+1,0] +
            w[-1,-1,0] + w[+1,+1,0] + w[-1,+1,0] + w[+1,-1,0] +

            w[-1,0,1] + w[+1,0,1] + w[0,-1,1] + w[0,+1,1] +
            w[-1,-1,1] + w[+1,+1,1] + w[-1,+1,1] + w[+1,-1,1] +
            w[0,0,1] +

            w[-1,0,-1] + w[+1,0,-1] + w[0,-1,-1] + w[0,+1,-1] +
            w[-1,-1,-1] + w[+1,+1,-1] + w[-1,+1,-1] + w[+1,-1,-1] +
            w[0,0,-1],1)


@numba.njit
def nanmean(v,w):
    return _nanmean(v,w)

filepath = '/net/so4/landclim/bverena/large_files/data_small.nc'

# open data
print(f'open data')
data = xr.open_dataarray(filepath)

# subset more for speedup of first tests
print(f'subset even more because very large dataset')
#data = data[:,::10,:,:]
shape = np.shape(data)

# create a mask of nans
mask = ~np.isnan(data) # nan values have zero weight (i.e. are False)

# gapfilling the missing values with spatiotemporal mean
print('gapfilling missing values with spatiotemporal mean')
tic = datetime.now()
result = _nanmean(data.values[0,:,:,:], mask.values[0,:,:,:])
result = _nanmean(data.values[1,:,:,:], mask.values[1,:,:,:])
result = _nanmean(data.values[2,:,:,:], mask.values[2,:,:,:])
toc = datetime.now()
print(f'this filter function took {toc-tic}')
tic = datetime.now() # slightly slower
result = nanmean(data.values[0,:,:,:], mask.values[0,:,:,:])
result = nanmean(data.values[1,:,:,:], mask.values[1,:,:,:])
result = nanmean(data.values[2,:,:,:], mask.values[2,:,:,:])
toc = datetime.now()
print(f'this filter function took {toc-tic}')
data = data.fillna(result)

# test if results are the same as in "ground truth"
from unittest_simple import test_simple
res = xr.open_dataarray('baseline_result.nc')
test_simple(data, res) # test fails bec one dim missing

# my PhD Project goes on with:
# gapfill each variable by regressing over all the others
# in an iterative EM-like fashion 
# with spatiotemporal gapfill as initial guess
# until estimates for missing values converge

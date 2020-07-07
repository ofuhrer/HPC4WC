"""
Example for workflow for gapfilling remote sensing data from diverse sources
Explicit stencil filter

    @author: verena bessenbacher
    @date: 12 06 2020
"""

import numpy as np
from datetime import datetime
import xarray as xr
from numba import jit


@jit
def local_nanmean(values):
    tmp = 0
    k = 0
    for v in values:
        if ~np.isnan(v):
            tmp = tmp + v
            k = k + 1
    if k != 0:
        return tmp / max(k,1)

@jit
def local_loop(data):
    shape = data.shape
    for t in range(2,shape[0]-2):
        print('new time t ='+str(t))
        for i in range(2,shape[1]-2):
            for j in range(2,shape[2]-2):
                data[t,i,j] = np.nanmean(data[t-2:t+3,i-2:i+3,j-2:j+3])

    return data


frac_missing = 0.42
filepath = '/net/so4/landclim/bverena/large_files/data_small.nc'

# create example array
print(f'open data array')
data = xr.open_dataarray(filepath)

# subset more for speedup of first tests
print(f'subset even more because very large dataset')
data = data[:,::10,:,:]

shape = np.shape(data)

# gapfilling the missing values with spatiotemporal mean
print('gapfilling missing values with spatiotemporal mean')
tic = datetime.now()
result = np.zeros(shape)
result[:,:,:,:] = np.nan
for var in range(0,shape[0]):
    import IPython; IPython.embed()
    data[var,:,:,:] = local_loop(data[var,:,:,:])


    #for t in range(2,shape[1]-2):
    #    print('new time t ='+str(t))
    #    for i in range(2,shape[2]-2):
    #        for j in range(2,shape[3]-2):
    #            #values = data[var,t-2:t+3,i-2:i+3,j-2:j+3]
    #            #values[2,2,2] = np.nan # changed this to nan so that it gets ignored
    #            #result[var,t,i,j] = nbnanmean(values, len(values), result[var,t,i,j], np.array([])) # DOES NOT WORK YET
    #            result[var,t,i,j] = local_nanmean(data[var,t-2:t+3,i-2:i+3,j-2:j+3].values.flatten())
toc = datetime.now()
print(f'this filter function took {toc-tic}')
data = data.fillna(result)

# test if results are the same as in "ground truth"
from unittest_simple import test_simple
res = xr.open_dataarray('baseline_result.nc')
import IPython; IPython.embed()
test_simple(data, res)

# my PhD Project goes on with:
# gapfill each variable by regressing over all the others
# in an iterative EM-like fashion 
# with spatiotemporal gapfill as initial guess
# until estimates for missing values converge

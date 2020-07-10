"""
Example for workflow for gapfilling remote sensing data from diverse sources
Explicit stencil filter

    @author: verena bessenbacher
    @date: 12 06 2020
"""

import numpy as np
from datetime import datetime
import xarray as xr
from scipy.ndimage.filters import generic_filter

shape = (3, 30, 72, 140) # real shape is (22, 3653, 720, 1440)
frac_missing = 0.42
filepath = '/net/so4/landclim/bverena/large_files/data_small.nc'

# create example array
print(f'create data array with shape {shape}')
data = xr.open_dataarray(filepath)

# subset more for speedup of first tests
print(f'subset even more because very large dataset')
data = data[:,::800,:,:]
data_orig = data.copy()

shape = np.shape(data)
# create a list of nan-indices
indices = np.where(np.isnan(data))

# gapfilling the missing values with spatiotemporal mean
print('gapfilling missing values with spatiotemporal mean')
tic = datetime.now()
result = np.zeros(shape)
result[:,:,:,:] = np.nan
# Alternatively do this with zip: https://stackoverflow.com/questions/21887138/iterate-over-the-output-of-np-where
for ind in range(0, len(indices[0])):
    var = indices[0][ind]
    t = indices[1][ind]
    i = indices[2][ind]
    j = indices[3][ind]
    # Ignore boundaries for now:
    if t in [0,1,shape[1]-1,shape[1]] or i in [0,1,shape[2]-1,shape[2]] or j in [0,1,shape[3]-1,shape[3]]:
        print('Skipping: '+str(t)+', '+str(i)+', '+str(j))
        continue
    print('Computing: '+str(t)+', '+str(i)+', '+str(j))
    tmp = 0
    k = 0
    values = data[var,t-2:t+3,i-2:i+3,j-2:j+3].copy()
    #print(np.shape(data), np.shape(values))
    values[2,2,2] = np.nan # changed this to nan so that it gets ignored
    values = values.values.flatten()
    for v in values:
        if ~np.isnan(v):
            tmp = tmp + v
            k = k + 1
    if k != 0:
        result[var,t,i,j] = tmp / max(k,1)
toc = datetime.now()
print(f'this filter function took {toc-tic}')
data = data.fillna(result)

# write to output
data.to_netcdf('baseline_iteratemask800.nc')

# test if results are the same as in "ground truth"
from unittest_simple import test_simple
res = xr.open_dataarray('baseline_result.nc')
import IPython; IPython.embed()
#test_simple(data, res)


# my PhD Project goes on with:
# gapfill each variable by regressing over all the others
# in an iterative EM-like fashion 
# with spatiotemporal gapfill as initial guess
# until estimates for missing values converge

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
data = data[:,::10,:,:]

shape = np.shape(data)
# create a mask of nans
data_mask = np.zeros(shape)
data_mask[np.where(np.isnan(data))] = 1

# gapfilling the missing values with spatiotemporal mean
print('gapfilling missing values with spatiotemporal mean')
tic = datetime.now()
result = np.zeros(shape)
result[:,:,:,:] = np.nan
for var in range(0,shape[0]):
    for t in range(2,shape[1]-2):
        print('new time t = '+str(t))
        for i in range(2,shape[2]-2):
            for j in range(2,shape[3]-2):
                if data_mask[var,t,i,j] == 0:
                    continue
                tmp = 0
                k = 0
                values = data[var,t-2:t+3,i-2:i+3,j-2:j+3]
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
data.to_netcdf('baseline_mask.nc')

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

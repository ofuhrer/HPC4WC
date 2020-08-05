"""
Example for workflow for gapfilling remote sensing data from diverse sources
Explicit stencil filter
Naive workflow

    @author: Ulrike Proske
    @date: 17 07 2020
"""

import numpy as np
from datetime import datetime
import xarray as xr
from scipy.ndimage.filters import generic_filter

shape = (3, 37, 72, 140) # real shape is (3, 3653, 720, 1440)
filepath = '/net/so4/landclim/bverena/large_files/data_small.nc'

# create example array
print(f'create data array with shape {shape}')
data = xr.open_dataarray(filepath)

# subset more for speedup of first tests
print(f'subset even more because very large dataset')
data = data[:,::100,:,:]

shape = np.shape(data)

# gapfilling the missing values with spatiotemporal mean
print('gapfilling missing values with spatiotemporal mean')
tic = datetime.now()
result = np.zeros(shape)
result[:,:,:,:] = np.nan
# loop over all data points
for var in range(0,shape[0]):
    for t in range(2,shape[1]-2):
        print('new time t = '+str(t))
        for i in range(2,shape[2]-2):
            for j in range(2,shape[3]-2):
                tmp = 0
                k = 0
                # copy box surrounding the data that is needed for the stencil
                values = data[var,t-2:t+3,i-2:i+3,j-2:j+3].copy()
                # flatten so that I can loop through the values
                values = values.values.flatten()
                for v in values:
                    # only data, no nans, shall be included
                    if ~np.isnan(v):
                        tmp = tmp + v
                        k = k + 1
                if k != 0:
                    # result = sum of data in box/ number of values in box
                    result[var,t,i,j] = tmp / max(k,1)
toc = datetime.now()
print(f'this filter function took {toc-tic}')
# only data points that are nan originally are filled with the filtered data
data = data.fillna(result)

# test if results are the same as in "ground truth"
from unittest_simple import test_simple
res = xr.open_dataarray('baseline_result.nc')
# debug: import IPython; IPython.embed()
test_simple(data, res)


# Verena's  PhD Project goes on with:
# gapfill each variable by regressing over all the others
# in an iterative EM-like fashion 
# with spatiotemporal gapfill as initial guess
# until estimates for missing values converge

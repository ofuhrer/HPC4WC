"""
Example for workflow for gapfilling remote sensing data from diverse sources

    @author: verena bessenbacher
    @date: 12 06 2020
"""

import numpy as np
from datetime import datetime
import xarray as xr
from scipy.ndimage.filters import generic_filter

frac_missing = 0.42
filepath = '/net/so4/landclim/bverena/large_files/data_small.nc'

# create example array
print(f'read data array')
data = xr.open_dataarray(filepath)

# subset more for speedup of first tests
print(f'subset even more because very large dataset')
data = data[:,::10,:,:]

# gapfilling the missing values with spatiotemporal mean
print('gapfilling missing values with spatiotemporal mean')
footprint = np.ones((1,5,5,5))
footprint[0,2,2,2] = 0 
tic = datetime.now()
tmp = generic_filter(data, np.nanmean, footprint=footprint, mode='nearest') # THIS IS SLOW!
toc = datetime.now()
print(f'this filter function took {toc-tic}')
data = data.fillna(tmp)

# save result as "ground truth" for testing other approaches
data.to_netcdf('baseline_result.nc')

# my PhD Project goes on with:
# gapfill each variable by regressing over all the others
# in an iterative EM-like fashion 
# with spatiotemporal gapfill as initial guess
# until estimates for missing values converge

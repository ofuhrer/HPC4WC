"""
Example for workflow for gapfilling remote sensing data from diverse sources

    @author: verena bessenbacher
    @date: 12 06 2020
"""

import numpy as np
from datetime import datetime
import xarray as xr
from scipy.ndimage.filters import generic_filter

shape = (3, 30, 72, 140) # real shape is (22, 3653, 720, 1440)
frac_missing = 0.42

# create example array
print(f'create data array with shape {shape}')
data = xr.DataArray(np.random.rand(*shape), dims=['variables', 'time', 'lat', 'lon'])

# real data has missing values, artificially introducing some
print(f'randomly deleting {frac_missing} percent of the data')
data.values[np.random.rand(*data.shape) < frac_missing] = np.nan

# gapfilling the missing values with spatiotemporal mean
print('gapfilling missing values with spatiotemporal mean')
footprint = np.ones((1,5,5,5))
footprint[0,2,2,2] = 0 
tic = datetime.now()
tmp = generic_filter(data, np.nanmean, footprint=footprint, mode='nearest') # THIS IS SLOW!
toc = datetime.now()
print(f'this filter function took {toc-tic}')
data = data.fillna(tmp)

# my PhD Project goes on with:
# gapfill each variable by regressing over all the others
# in an iterative EM-like fashion 
# with spatiotemporal gapfill as initial guess
# until estimates for missing values converge

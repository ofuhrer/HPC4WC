"""
Example for workflow for gapfilling remote sensing data from diverse sources

    @author: verena bessenbacher
    @date: 12 06 2020
"""

import numpy as np
import xarray as xr
from scipy.ndimage.filters import generic_filter
from sklearn.utils.random import sample_without_replacement

shape = (3, 30, 72, 140) # real shape is (22, 3653, 720, 1440)
frac_missing = 0.42

# create example array
print(f'create data array with shape {shape}')
data = xr.DataArray(np.random.rand(*shape), dims=['variables', 'time', 'lat', 'lon'])

# real data has missing values, artificially introducing some
print(f'randomly deleting {frac_missing} percent of the data')
n_samples = data.size * frac_missing
idxs = sample_without_replacement(data.size, n_samples)
data.values.flat[idxs] = np.nan

# gapfilling the missing values with spatiotemporal mean
print('gapfilling missing values with spatiotemporal mean')
footprint = np.ones((1,5,5,5))
footprint[0,2,2,2] = 0 
tmp = generic_filter(data, np.nanmean, footprint=footprint, mode='nearest') # THIS IS SLOW!
data = data.fillna(tmp)

# my PhD Project goes on with:
# gapfill each variable by regressing over all the others
# in an iterative EM-like fashion 
# with spatiotemporal gapfill as initial guess
# until estimates for missing values converge

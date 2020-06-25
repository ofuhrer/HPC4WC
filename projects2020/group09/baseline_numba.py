"""
Example for workflow for gapfilling remote sensing data from diverse sources

    @author: verena bessenbacher
    @date: 12 06 2020
"""

import numpy as np
from datetime import datetime
import xarray as xr
from scipy.ndimage.filters import generic_filter
from numba_nanmean import nbnanmean

shape = (3, 30, 72, 140) # real shape is (22, 3653, 720, 1440)
frac_missing = 0.42
filepath = '/net/so4/landclim/bverena/large_files/data_small.nc'

# create example array
print(f'create data array with shape {shape}')
data = xr.open_dataarray(filepath)

# subset more for speedup of first tests
print(f'subset even more because very large dataset')
data = data[:,::10,:,:]

# gapfilling the missing values with spatiotemporal mean
print('gapfilling missing values with spatiotemporal mean')
from scipy import LowLevelCallable                   
mean_fct = LowLevelCallable(nbnanmean.ctypes) 
footprint = np.ones((1,5,5,5))
footprint[0,2,2,2] = 0 
tic = datetime.now()
tmp = generic_filter(data, mean_fct, footprint=footprint, mode='nearest') # THIS IS SLOW!
toc = datetime.now()
print(f'this filter function took {toc-tic}')
data = data.fillna(tmp)

# my PhD Project goes on with:
# gapfill each variable by regressing over all the others
# in an iterative EM-like fashion 
# with spatiotemporal gapfill as initial guess
# until estimates for missing values converge

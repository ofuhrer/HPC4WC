"""
Test for the fortran script for gapfilling remote sensing data from diverse sources
This test is slow, but working (I only use it for small example fields).

    @author: Ulrike Proske
    @date: 17 07 2020
"""

import numpy as np
import numba
from datetime import datetime
import xarray as xr
from scipy.ndimage.filters import generic_filter

# absolute tolerance for the isclose validation:
abstol = 0.05
filepath = './foin.nc'

# create example array
print(f'read data array')
data = xr.open_dataarray(filepath)
data = data[:,:,:,:]
print(data.shape)
for i in range(0,np.shape(data)[0]):
    print(i)
    for j in range(0,np.shape(data)[1]):
        for k in range(0,np.shape(data)[2]):
            for h in range(0,np.shape(data)[3]):
                if data[i,j,k,h] == 0:
                    data[i,j,k,h] = np.nan

# numpy 
tic = datetime.now()
footprint = np.ones((5,5,5,1))
tmp = generic_filter(data, np.nanmean, footprint=footprint, mode='nearest')
toc = datetime.now()
print(f'numpy {toc-tic}')
result = xr.open_dataarray('./fout.nc')
result = result[:,:,:,:]
# debug import IPython; IPython.embed()
# where tmp is nan, this should be changed back to 0, because that's what the fortran code deals with instead of nans:
for i in range(0,np.shape(tmp)[0]):
    print(i)
    for j in range(0,np.shape(tmp)[1]):
        for k in range(0,np.shape(tmp)[2]):
            for h in range(0,np.shape(tmp)[3]):
                if np.isnan(tmp[i,j,k,h]):
                    tmp[i,j,k,h] = 0

vali = np.isclose(result[2:-2,2:-2,2:-2,:], tmp[2:-2,2:-2,2:-2,:], atol=abstol, equal_nan=True).all()
print(f'validated: {vali}, absolute tolerance: {abstol}')
vali = np.isclose(result[2:-2,2:-2,2:-2,:], tmp[2:-2,2:-2,2:-2,:], equal_nan=True).all()
print(f'validated: {vali}, without specified absolute tolerance')
#import IPython; IPython.embed()


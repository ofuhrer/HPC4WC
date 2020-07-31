"""
Halo update functions for all python gapfilling routines

    @author: Ulrike Proske
    @date: 31 07 2020
"""

import numpy as np
import numba
from datetime import datetime
import xarray as xr
from scipy.ndimage.filters import generic_filter

def halo_update(data, tmp):
    for ivar in range(0,np.shape(data)[0]):
        # sides/edges
        for i in range(2, np.shape(data)[1]-2): # exclude corners
            for j in range(2, np.shape(data)[2]-2):
                    # left boundary:
                    # left 0
                    box = data[ivar, i-2:i+3, j-2:j+3, -2:3] # var, time, lat, lon(-2,-1,0,1,2)
                    print(np.shape(box))
                    import IPython; IPython.embed()
                    tmp[ivar,i,j,0] = np.nanmean(data[ivar, i-2:i+3, j-2:j+3, -2:3])
                    # left 1
                    box = data[ivar, i-2:i+3, j-2:j+3, -1:4] # var, time, lat, lon(-1,0,1,2,3)
                    print(np.shape(box))
                    tmp[ivar,i,j,1] = np.nanmean(data[ivar, i-2:i+3, j-2:j+3, -1:4])

                    # right boundary:
                    # right -2
                    box = data[ivar, i-2:i+3, j-2:j+3, -4:1] # var, time, lat, lon(-4,-3,-2,-1,0)
                    print(np.shape(box))
                    tmp[ivar,i,j,-2] = np.nanmean(data[ivar, i-2:i+3, j-2:j+3, -4:1])
                    # right -1
                    box = data[ivar, i-2:i+3, j-2:j+3, -3:2] # var, time, lat, lon(-3,-2,-1,0,1)
                    print(np.shape(box))
                    tmp[ivar,i,j,-1] = np.nanmean(data[ivar, i-2:i+3, j-2:j+3, -3:2])
        # top and bottom
        for i in range(2, np.shape(data)[1]-2):
            for k in range(2, np.shape(data)[3]-2):
                    # lower boundary:
                    # bottom 0
                    box = data[ivar, i-2:i+3, 0:3, k-2:k+3] # simply choose a smaller box
                    print(np.shape(box))
                    tmp[ivar,i,0,k] = np.nanmean(data[ivar, i-2:i+3, 0:3, k-2:k+3])
                    # bottom 1
                    box = data[ivar, i-2:i+3, 0:4, k-2:k+3] # simply choose a smaller box
                    print(np.shape(box))
                    tmp[ivar,i,1,k] = np.nanmean(data[ivar, i-2:i+3, 0:4, k-2:k+3])
                    # upper boundary:
                    # top -1
                    box = data[ivar, i-2:i+3, -3:0, k-2:k+3] # simply choose a smaller box
                    print(np.shape(box))
                    tmp[ivar,i,-1,k] = np.nanmean(data[ivar, i-2:i+3, -3:0, k-2:k+3])
                    # top -2
                    box = data[ivar, i-2:i+3, -4:0, k-2:k+3] # simply choose a smaller box
                    print(np.shape(box))
                    tmp[ivar,i,-2,k] = np.nanmean(data[ivar, i-2:i+3, -4:0, k-2:k+3])
                    # TODO: write these function calls more elegantly, like for time
        # time
        for j in range(2, np.shape(data)[2]-2):
            for k in range(2, np.shape(data)[3]-2):
                # time boundaries don't make sense -> simply choose a smaller box
                    # beginning
                    # beginning 0
                    tmp[ivar,0,j,k] = np.nanmean(data[ivar,0:3,j-2:j+2,k-2:k+2])
                    # beginning 1
                    tmp[ivar,1,j,k] = np.nanmean(data[ivar,0:4,j-2:j+2,k-2:k+2])
                    # end
                    # end -1
                    tmp[ivar,-1,j,k] = np.nanmean(data[ivar,-3:0,j-2:j+2,k-2:k+2])
                    # end -2
                    tmp[ivar,-2,j,k] = np.nanmean(data[ivar,-4:0,j-2:j+2,k-2:k+2])
        # corners
        # 4 corners in time=0-level
        tmp[ivar,0,0,0] = np.nanmean(data[ivar,0:3,0:3,-2:3])
        tmp[ivar,0,0,1] = np.nanmean(data[ivar,0:3,0:3,-1:4])
        tmp[ivar,0,1,0] = np.nanmean(data[ivar,0:3,0:4,-2:3])
        tmp[ivar,0,1,1] = np.nanmean(data[ivar,0:3,0:4,-1:4])
        tmp[ivar,0,-2,0] = np.nanmean(data[ivar,0:3,-4:0,-2:3])
        tmp[ivar,0,-2,1] = np.nanmean(data[ivar,0:3,-4:0,-1:4])
        tmp[ivar,0,-1,0] = np.nanmean(data[ivar,0:3,-3:0,-2:3])
        tmp[ivar,0,-1,1] = np.nanmean(data[ivar,0:3,-3:0,-1:4])
        tmp[ivar,0,0,-2] = np.nanmean(data[ivar,0:3,0:3,-4:1])
        tmp[ivar,0,0,-1] = np.nanmean(data[ivar,0:3,0:3,-3:2])
        tmp[ivar,0,1,-2] = np.nanmean(data[ivar,0:3,0:4,-4:1])
        tmp[ivar,0,1,-1] = np.nanmean(data[ivar,0:3,0:4,-3:2])
        tmp[ivar,0,-2,-2] = np.nanmean(data[ivar,0:3,-4:0,-4:1])
        tmp[ivar,0,-2,-1] = np.nanmean(data[ivar,0:3,-4:0,-3:2])
        tmp[ivar,0,-1,-2] = np.nanmean(data[ivar,0:3,-3:0,-4:1])
        tmp[ivar,0,-1,-1] = np.nanmean(data[ivar,0:3,-3:0,-4:1])
        # 4 corners in time=1-level
        tmp[ivar,1,0,0] = np.nanmean(data[ivar,0:4,0:3,-2:3])
        tmp[ivar,1,0,1] = np.nanmean(data[ivar,0:4,0:3,-1:4])
        tmp[ivar,1,1,0] = np.nanmean(data[ivar,0:4,0:4,-2:3])
        tmp[ivar,1,1,1] = np.nanmean(data[ivar,0:4,0:4,-1:4])
        tmp[ivar,1,-2,0] = np.nanmean(data[ivar,0:4,-4:0,-2:3])
        tmp[ivar,1,-2,1] = np.nanmean(data[ivar,0:4,-4:0,-1:4])
        tmp[ivar,1,-1,0] = np.nanmean(data[ivar,0:4,-3:0,-2:3])
        tmp[ivar,1,-1,1] = np.nanmean(data[ivar,0:4,-3:0,-1:4])
        tmp[ivar,1,0,-2] = np.nanmean(data[ivar,0:4,0:3,-4:1])
        tmp[ivar,1,0,-1] = np.nanmean(data[ivar,0:4,0:3,-3:2])
        tmp[ivar,1,1,-2] = np.nanmean(data[ivar,0:4,0:4,-4:1])
        tmp[ivar,1,1,-1] = np.nanmean(data[ivar,0:4,0:4,-3:2])
        tmp[ivar,1,-2,-2] = np.nanmean(data[ivar,0:4,-4:0,-4:1])
        tmp[ivar,1,-2,-1] = np.nanmean(data[ivar,0:4,-4:0,-3:2])
        tmp[ivar,1,-1,-2] = np.nanmean(data[ivar,0:4,-3:0,-4:1])
        tmp[ivar,1,-1,-1] = np.nanmean(data[ivar,0:4,-3:0,-4:1])
        # 4 corners in time=-1-level
        tmp[ivar,-1,0,0] = np.nanmean(data[ivar,-3:1,0:3,-2:3])
        tmp[ivar,-1,0,1] = np.nanmean(data[ivar,-3:1,0:3,-1:4])
        tmp[ivar,-1,1,0] = np.nanmean(data[ivar,-3:1,0:4,-2:3])
        tmp[ivar,-1,1,1] = np.nanmean(data[ivar,-3:1,0:4,-1:4])
        tmp[ivar,-1,-2,0] = np.nanmean(data[ivar,-3:1,-4:0,-2:3])
        tmp[ivar,-1,-2,1] = np.nanmean(data[ivar,-3:1,-4:0,-1:4])
        tmp[ivar,-1,-1,0] = np.nanmean(data[ivar,-3:1,-3:0,-2:3])
        tmp[ivar,-1,-1,1] = np.nanmean(data[ivar,-3:1,-3:0,-1:4])
        tmp[ivar,-1,0,-2] = np.nanmean(data[ivar,-3:1,0:3,-4:1])
        tmp[ivar,-1,0,-1] = np.nanmean(data[ivar,-3:1,0:3,-3:2])
        tmp[ivar,-1,1,-2] = np.nanmean(data[ivar,-3:1,0:4,-4:1])
        tmp[ivar,-1,1,-1] = np.nanmean(data[ivar,-3:1,0:4,-3:2])
        tmp[ivar,-1,-2,-2] = np.nanmean(data[ivar,-3:1,-4:0,-4:1])
        tmp[ivar,-1,-2,-1] = np.nanmean(data[ivar,-3:1,-4:0,-3:2])
        tmp[ivar,-1,-1,-2] = np.nanmean(data[ivar,-3:1,-3:0,-4:1])
        tmp[ivar,-1,-1,-1] = np.nanmean(data[ivar,-3:1,-3:0,-4:1])
        # 4 corners in time=-2-level
        tmp[ivar,-2,0,0] = np.nanmean(data[ivar,-4:1,0:3,-2:3])
        tmp[ivar,-2,0,1] = np.nanmean(data[ivar,-4:1,0:3,-1:4])
        tmp[ivar,-2,1,0] = np.nanmean(data[ivar,-4:1,0:4,-2:3])
        tmp[ivar,-2,1,1] = np.nanmean(data[ivar,-4:1,0:4,-1:4])
        tmp[ivar,-2,-2,0] = np.nanmean(data[ivar,-4:1,-4:0,-2:3])
        tmp[ivar,-2,-2,1] = np.nanmean(data[ivar,-4:1,-4:0,-1:4])
        tmp[ivar,-2,-1,0] = np.nanmean(data[ivar,-4:1,-3:0,-2:3])
        tmp[ivar,-2,-1,1] = np.nanmean(data[ivar,-4:1,-3:0,-1:4])
        tmp[ivar,-2,0,-2] = np.nanmean(data[ivar,-4:1,0:3,-4:1])
        tmp[ivar,-2,0,-1] = np.nanmean(data[ivar,-4:1,0:3,-3:2])
        tmp[ivar,-2,1,-2] = np.nanmean(data[ivar,-4:1,0:4,-4:1])
        tmp[ivar,-2,1,-1] = np.nanmean(data[ivar,-4:1,0:4,-3:2])
        tmp[ivar,-2,-2,-2] = np.nanmean(data[ivar,-4:1,-4:0,-4:1])
        tmp[ivar,-2,-2,-1] = np.nanmean(data[ivar,-4:1,-4:0,-3:2])
        tmp[ivar,-2,-1,-2] = np.nanmean(data[ivar,-4:1,-3:0,-4:1])
        tmp[ivar,-2,-1,-1] = np.nanmean(data[ivar,-4:1,-3:0,-4:1])
    return


frac_missing = 0.42
filepath = '/net/so4/landclim/bverena/large_files/data_small.nc'

# create example array
print(f'read data array')
data = xr.open_dataarray(filepath)

# subset more for speedup of first tests
print(f'subset even more because very large dataset')
data = data[:,::100,:,:]
print(data.shape)

# numpy 
tic = datetime.now()
footprint = np.ones((1,5,5,5))
#tmp = generic_filter(data, np.nanmean, footprint=footprint, mode='nearest')
toc = datetime.now()
print(f'numpy {toc-tic}')

# numba njit
@numba.njit
def numba_nanmean(values):
    return np.nanmean(values)
tic = datetime.now()
footprint = np.ones((1,3,3,3))
tmp = generic_filter(data, numba_nanmean, footprint=footprint, mode='nearest')
halo_update(data,tmp)
toc = datetime.now()
print(f'numba {toc-tic}')

"""
# unnecessary, ignore

def halo_stencil(box):

    
def halo_stencil_l3(v,w):
        # careful, l3 and l4 have different shapes for time and lat
        if np.shape(v)[0] == 3: # time
            print('time l3')
            return v[0,0,0] * w[0,0,0] +
                   v[0,0,1] * w[0,0,1] + 
                   v[0,0,2] * w[0,0,2] +
                   v[0,0,3] * w[0,0,3] +
                   v[0,0,4] * w[0,0,4] +
                   v[0,1,0] * w[0,1,0] +
                   v[0,1,1] * w[0,1,1] +
                   v[0,1,2] * w[0,1,2] +
                   v[0,1,3] * w[0,1,3] +
                   v[0,1,4] * w[0,1,4] +
                   v[0,2,0] * w[0,2,0] +
                   v[0,2,1] * w[0,2,1] +
                   v[0,2,2] * w[0,2,2] +
                   v[0,2,3] * w[0,2,3] +
                   v[0,2,4] * w[0,2,4] +
                   v[0,3,0] * w[0,3,0] +
                   v[0,3,1] * w[0,3,1] +
                   v[0,3,2] * w[0,3,2] +
                   v[0,3,3] * w[0,3,3] +
                   v[0,3,4] * w[0,3,4] +
                   v[0,4,0] * w[0,4,0] +
                   v[0,4,1] * w[0,4,1] +
                   v[0,4,2] * w[0,4,2] +
                   v[0,4,3] * w[0,4,3] +
                   v[0,4,4] * w[0,4,4] +
                   v[1,0,0] * w[1,0,0] +
                   v[1,0,1] * w[1,0,1] +
                   v[1,0,2] * w[1,0,2] +
                   v[1,0,3] * w[1,0,3] +
                   v[1,0,4] * w[1,0,4] +
                   v[1,1,0] * w[1,1,0] +
                   v[1,1,1] * w[1,1,1] +
                   v[1,1,2] * w[1,1,2] +
                   v[1,1,3] * w[1,1,3] +
                   v[1,1,4] * w[1,1,4] +
                   v[1,2,0] * w[1,2,0] +
                   v[1,2,1] * w[1,2,1] +
                   v[1,2,2] * w[1,2,2] +
                   v[1,2,3] * w[1,2,3] +
                   v[1,2,4] * w[1,2,4] +
                   v[1,3,0] * w[1,3,0] +
                   v[1,3,1] * w[1,3,1] +
                   v[1,3,2] * w[1,3,2] +
                   v[1,3,3] * w[1,3,3] +
                   v[1,3,4] * w[1,3,4] +
                   v[1,4,0] * w[1,4,0] +
                   v[1,4,1] * w[1,4,1] +
                   v[1,4,2] * w[1,4,2] +
                   v[1,4,3] * w[1,4,3] +
                   v[1,4,4] * w[1,4,4] +
                   v[2,0,0] * w[2,0,0] +
                   v[2,0,1] * w[2,0,1] +
                   v[2,0,2] * w[2,0,2] +
                   v[2,0,3] * w[2,0,3] +
                   v[2,0,4] * w[2,0,4] +
                   v[2,1,0] * w[2,1,0] +
                   v[2,1,1] * w[2,1,1] +
                   v[2,1,2] * w[2,1,2] +
                   v[2,1,3] * w[2,1,3] +
                   v[2,1,4] * w[2,1,4] +
                   v[2,2,0] * w[2,2,0] +
                   v[2,2,1] * w[2,2,1] +
                   v[2,2,2] * w[2,2,2] +
                   v[2,2,3] * w[2,2,3] +
                   v[2,2,4] * w[2,2,4] +
                   v[2,3,0] * w[2,3,0] +
                   v[2,3,1] * w[2,3,1] +
                   v[2,3,2] * w[2,3,2] +
                   v[2,3,3] * w[2,3,3] +
                   v[2,3,4] * w[2,3,4] +
                   v[2,4,0] * w[2,4,0] +
                   v[2,4,1] * w[2,4,1] +
                   v[2,4,2] * w[2,4,2] +
                   v[2,4,3] * w[2,4,3] +
                   v[2,4,4] * w[2,4,4]
        if np.shape(v)[1] == 3: # lat
            print('time l3')
            return v[0,0,0] * w[0,0,0] +
                   v[0,0,1] * w[0,0,1] + 
                   v[0,0,2] * w[0,0,2] +
                   v[0,0,3] * w[0,0,3] +
                   v[0,0,4] * w[0,0,4] +
                   v[0,1,0] * w[0,1,0] +
                   v[0,1,1] * w[0,1,1] +
                   v[0,1,2] * w[0,1,2] +
                   v[0,1,3] * w[0,1,3] +
                   v[0,1,4] * w[0,1,4] +
                   v[0,2,0] * w[0,2,0] +
                   v[0,2,1] * w[0,2,1] +
                   v[0,2,2] * w[0,2,2] +
                   v[0,2,3] * w[0,2,3] +
                   v[0,2,4] * w[0,2,4] +
                   v[1,0,0] * w[1,0,0] +
                   v[1,0,1] * w[1,0,1] +
                   v[1,0,2] * w[1,0,2] +
                   v[1,0,3] * w[1,0,3] +
                   v[1,0,4] * w[1,0,4] +
                   v[1,1,0] * w[1,1,0] +
                   v[1,1,1] * w[1,1,1] +
                   v[1,1,2] * w[1,1,2] +
                   v[1,1,3] * w[1,1,3] +
                   v[1,1,4] * w[1,1,4] +
                   v[1,2,0] * w[1,2,0] +
                   v[1,2,1] * w[1,2,1] +
                   v[1,2,2] * w[1,2,2] +
                   v[1,2,3] * w[1,2,3] +
                   v[1,2,4] * w[1,2,4] +
                   v[1,3,0] * w[1,3,0] +
                   v[2,0,0] * w[2,0,0] +
                   v[2,0,1] * w[2,0,1] +
                   v[2,0,2] * w[2,0,2] +
                   v[2,0,3] * w[2,0,3] +
                   v[2,0,4] * w[2,0,4] +
                   v[2,1,0] * w[2,1,0] +
                   v[2,1,1] * w[2,1,1] +
                   v[2,1,2] * w[2,1,2] +
                   v[2,1,3] * w[2,1,3] +
                   v[2,1,4] * w[2,1,4] +
                   v[2,2,0] * w[2,2,0] +
                   v[2,2,1] * w[2,2,1] +
                   v[2,2,2] * w[2,2,2] +
                   v[2,2,3] * w[2,2,3] +
                   v[2,2,4] * w[2,2,4] +
                   v[3,0,0] * w[3,0,0] +
                   v[3,0,1] * w[3,0,1] +
                   v[3,0,2] * w[3,0,2] +
                   v[3,0,3] * w[3,0,3] +
                   v[3,0,4] * w[3,0,4] +
                   v[3,1,0] * w[3,1,0] +
                   v[3,1,1] * w[3,1,1] +
                   v[3,1,2] * w[3,1,2] +
                   v[3,1,3] * w[3,1,3] +
                   v[3,1,4] * w[3,1,4] +
                   v[3,2,0] * w[3,2,0] +
                   v[3,2,1] * w[3,2,1] +
                   v[3,2,2] * w[3,2,2] +
                   v[3,2,3] * w[3,2,3] +
                   v[3,2,4] * w[3,2,4] +
                   v[4,0,0] * w[4,0,0] +
                   v[4,0,1] * w[4,0,1] +
                   v[4,0,2] * w[4,0,2] +
                   v[4,0,3] * w[4,0,3] +
                   v[4,0,4] * w[4,0,4] +
                   v[4,1,0] * w[4,1,0] +
                   v[4,1,1] * w[4,1,1] +
                   v[4,1,2] * w[4,1,2] +
                   v[4,1,3] * w[4,1,3] +
                   v[4,1,4] * w[4,1,4] +
                   v[4,2,0] * w[4,2,0] +
                   v[4,2,1] * w[4,2,1] +
                   v[4,2,2] * w[4,2,2] +
                   v[4,2,3] * w[4,2,3] +
                   v[4,2,4] * w[4,2,4]
"""

# numba stencil
@numba.stencil
def _sum(w):
    return (w[-1,0,0] + w[+1,0,0] + w[0,-1,0] + w[0,+1,0] +
            w[-1,-1,0] + w[+1,+1,0] + w[-1,+1,0] + w[+1,-1,0] + w[0,0,0] +

            w[-1,0,1] + w[+1,0,1] + w[0,-1,1] + w[0,+1,1] +
            w[-1,-1,1] + w[+1,+1,1] + w[-1,+1,1] + w[+1,-1,1] + w[0,0,1] +

            w[-1,0,-1] + w[+1,0,-1] + w[0,-1,-1] + w[0,+1,-1] +
            w[-1,-1,-1] + w[+1,+1,-1] + w[-1,+1,-1] + w[+1,-1,-1] + w[0,0,-1])

mask = ~ np.isnan(data.values)
datanum = np.nan_to_num(data)
tic = datetime.now()
result = np.empty(data.shape)
weights = np.empty(data.shape)
result[0,:,:,:] = _sum(datanum[0,:,:,:])
result[1,:,:,:] = _sum(datanum[1,:,:,:])
result[2,:,:,:] = _sum(datanum[2,:,:,:])
weights[0,:,:,:] = _sum(mask[0,:,:,:])
weights[1,:,:,:] = _sum(mask[1,:,:,:])
weights[2,:,:,:] = _sum(mask[2,:,:,:])
result = result / weights
result = np.where(weights == 0, np.nan, result)
toc = datetime.now()
vali = np.isclose(result[:,2:-2,2:-2,2:-2], tmp[:,2:-2,2:-2,2:-2], equal_nan=True).all()
print(f'stencil {toc-tic}, validated: {vali}')

# numba stencil and njit
@numba.njit
def numba_sum(w):
    return _sum(w)

tic = datetime.now()
result = np.empty(data.shape)
weights = np.empty(data.shape)
result[0,:,:,:] = numba_sum(datanum[0,:,:,:])
result[1,:,:,:] = numba_sum(datanum[1,:,:,:])
result[2,:,:,:] = numba_sum(datanum[2,:,:,:])
weights[0,:,:,:] = numba_sum(mask[0,:,:,:])
weights[1,:,:,:] = numba_sum(mask[1,:,:,:])
weights[2,:,:,:] = numba_sum(mask[2,:,:,:])
result = result / weights
result = np.where(weights == 0, np.nan, result)
toc = datetime.now()
vali = np.isclose(result[:,2:-2,2:-2,2:-2], tmp[:,2:-2,2:-2,2:-2], equal_nan=True).all()
print(f'stencil + njit {toc-tic}, validated: {vali}')

# cython stencil
from cython_loop import stencil_loop
mask = (~ np.isnan(data.values) *1.)
mask = mask.astype(np.float32)
tic = datetime.now()
result = np.empty(data.shape, dtype=np.float32)
weights = np.empty(data.shape, dtype=np.float32)
result[0,:,:,:] = stencil_loop(datanum[0,:,:,:])
result[1,:,:,:] = stencil_loop(datanum[1,:,:,:])
result[2,:,:,:] = stencil_loop(datanum[2,:,:,:])
weights[0,:,:,:] = stencil_loop(mask[0,:,:,:])
weights[1,:,:,:] = stencil_loop(mask[1,:,:,:])
weights[2,:,:,:] = stencil_loop(mask[2,:,:,:])
result = result / weights
result = np.where(weights == 0, np.nan, result)
toc = datetime.now()
vali = np.isclose(result[:,2:-2,2:-2,2:-2], tmp[:,2:-2,2:-2,2:-2], equal_nan=True).all()
print(f'cython {toc-tic}, validated: {vali}')

# cython stencil
from cython_loop import stencil_loop_blocking as stencil_loop
mask = (~ np.isnan(data.values) *1.)
mask = mask.astype(np.float32)
tic = datetime.now()
result = np.empty(data.shape, dtype=np.float32)
weights = np.empty(data.shape, dtype=np.float32)
result[0,:,:,:] = stencil_loop(datanum[0,:,:,:])
result[1,:,:,:] = stencil_loop(datanum[1,:,:,:])
result[2,:,:,:] = stencil_loop(datanum[2,:,:,:])
weights[0,:,:,:] = stencil_loop(mask[0,:,:,:])
weights[1,:,:,:] = stencil_loop(mask[1,:,:,:])
weights[2,:,:,:] = stencil_loop(mask[2,:,:,:])
result = result / weights
result = np.where(weights == 0, np.nan, result)
toc = datetime.now()
vali = np.isclose(result[:,2:-2,2:-2,2:-2], tmp[:,2:-2,2:-2,2:-2], equal_nan=True).all()
print(f'cython block {toc-tic}, validated: {vali}')

# stencil debugging area
nx = 11 # total size
ny = 11
nz = 11
blockx = 5 # block size
blocky = 5
blockz = 5
blocki = nx//blockx+1 # number of blocks (last one smaller)
blockj = ny//blocky+1
blockk = nz//blockz+1

blocksizes_x = [blockx] * (nx//blockx) + [nx%blockx]
blocksizes_y = [blocky] * (ny//blocky) + [ny%blocky]
blocksizes_z = [blockz] * (nz//blockz) + [nz%blockz]

A = np.zeros((nx,ny,nz))
for iblock, iblocklen in enumerate(blocksizes_x):
    for jblock, jblocklen in enumerate(blocksizes_y):
        for kblock, kblocklen in enumerate(blocksizes_z):
            for i_local in range(iblocklen):
                for j_local in range(jblocklen):
                    for k_local in range(kblocklen):
                        i = iblock*blockx + i_local
                        j = jblock*blocky + j_local
                        k = kblock*blockz + k_local
                        A[i,j,k] = 1

A = np.zeros((nx,ny,nz))
for iblock in range(blocki):
    if iblock == blocki-1: # we are in the last block
        iblocklen = nx%blockx
    else: # we are not in the last block
        iblocklen = blockx
    for jblock in range(blockj):
        if jblock == blockj-1: # we are in the last block
            jblocklen = ny%blocky
        else: # we are not in the last block
            jblocklen = blocky
        #print(jblock, blockj, jblocklen)
        for kblock in range(blockk):
            if kblock == blockk-1: # we are in the last block
                kblocklen = nz%blockz
            else: # we are not in the last block
                kblocklen = blockz
            for i_local in range(iblocklen):
                for j_local in range(jblocklen):
                    for k_local in range(kblocklen):
                        i = iblock*blockx + i_local
                        j = jblock*blocky + j_local
                        k = kblock*blockz + k_local
                        A[i,j,k] = 1
#data = data.fillna(tmp)
# save result as "ground truth" for testing other approaches
#data.to_netcdf('baseline_result.nc')
# my PhD Project goes on with:
# gapfill each variable by regressing over all the others
# in an iterative EM-like fashion 
# with spatiotemporal gapfill as initial guess
# until estimates for missing values converge

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

# numba njit
@numba.njit
def numba_nanmean(values):
    return np.nanmean(values)


def halo_update(data, tmp):
    for ivar in range(0,np.shape(data)[0]):
        # sides/edges
        for i in range(2, np.shape(data)[1]-2): # exclude corners
            for j in range(2, np.shape(data)[2]-2):
                    # left boundary:
                    # left 0
                    # var, time, lat, lon(-2,-1,0,1,2)
                    #print(np.shape(box))
                    tmp[ivar,i,j,0] = numba_nanmean(data[ivar, i-2:i+3, j-2:j+3, np.r_[-2:3]].values)
                    # left 1
                    # var, time, lat, lon(-1,0,1,2,3)
                    tmp[ivar,i,j,1] = numba_nanmean(data[ivar, i-2:i+3, j-2:j+3, np.r_[-1:4]].values)

                    # right boundary:
                    # right -2
                    # var, time, lat, lon(-4,-3,-2,-1,0)
                    tmp[ivar,i,j,-2] = numba_nanmean(data[ivar, i-2:i+3, j-2:j+3, np.r_[-4:1]].values)
                    # right -1
                    # var, time, lat, lon(-3,-2,-1,0,1)
                    tmp[ivar,i,j,-1] = numba_nanmean(data[ivar, i-2:i+3, j-2:j+3, np.r_[-3:2]].values)
        # top and bottom
        for i in range(2, np.shape(data)[1]-2):
            for k in range(2, np.shape(data)[3]-2):
                    # lower boundary:
                    # bottom 0
                    # simply choose a smaller box
                    tmp[ivar,i,0,k] = numba_nanmean(data[ivar, i-2:i+3, 0:3, k-2:k+3].values)
                    # bottom 1
                    # simply choose a smaller box
                    tmp[ivar,i,1,k] = numba_nanmean(data[ivar, i-2:i+3, 0:4, k-2:k+3].values)
                    # upper boundary:
                    # top -1
                    # simply choose a smaller box
                    tmp[ivar,i,-1,k] = numba_nanmean(data[ivar, i-2:i+3, np.r_[-3:0], k-2:k+3].values)
                    # top -2
                    # simply choose a smaller box
                    tmp[ivar,i,-2,k] = numba_nanmean(data[ivar, i-2:i+3, np.r_[-4:0], k-2:k+3].values)
        # time
        for j in range(2, np.shape(data)[2]-2):
            for k in range(2, np.shape(data)[3]-2):
                # time boundaries don't make sense -> simply choose a smaller box
                    # beginning
                    # beginning 0
                    tmp[ivar,0,j,k] = numba_nanmean(data[ivar,0:3,j-2:j+2,k-2:k+2].values)
                    # beginning 1
                    tmp[ivar,1,j,k] = numba_nanmean(data[ivar,0:4,j-2:j+2,k-2:k+2].values)
                    # end
                    # end -1
                    tmp[ivar,-1,j,k] = numba_nanmean(data[ivar,np.r_[-3:0],j-2:j+2,k-2:k+2].values)
                    # end -2
                    tmp[ivar,-2,j,k] = numba_nanmean(data[ivar,np.r_[-4:0],j-2:j+2,k-2:k+2].values)
        # corners
        # 4 corners in time=0-level
        tmp[ivar,0,0,0] = numba_nanmean(data[ivar,0:3,0:3,np.r_[-2:3]].values)
        tmp[ivar,0,0,1] = numba_nanmean(data[ivar,0:3,0:3,np.r_[-1:4]].values)
        tmp[ivar,0,1,0] = numba_nanmean(data[ivar,0:3,0:4,np.r_[-2:3]].values)
        tmp[ivar,0,1,1] = numba_nanmean(data[ivar,0:3,0:4,np.r_[-1:4]].values)
        tmp[ivar,0,-2,0] = numba_nanmean(data[ivar,0:3,np.r_[-4:0],np.r_[-2:3]].values)
        tmp[ivar,0,-2,1] = numba_nanmean(data[ivar,0:3,np.r_[-4:0],np.r_[-1:4]].values)
        tmp[ivar,0,-1,0] = numba_nanmean(data[ivar,0:3,np.r_[-3:0],np.r_[-2:3]].values)
        tmp[ivar,0,-1,1] = numba_nanmean(data[ivar,0:3,np.r_[-3:0],np.r_[-1:4]].values)
        tmp[ivar,0,0,-2] = numba_nanmean(data[ivar,0:3,0:3,np.r_[-4:1]].values)
        tmp[ivar,0,0,-1] = numba_nanmean(data[ivar,0:3,0:3,np.r_[-3:2]].values)
        tmp[ivar,0,1,-2] = numba_nanmean(data[ivar,0:3,0:4,np.r_[-4:1]].values)
        tmp[ivar,0,1,-1] = numba_nanmean(data[ivar,0:3,0:4,np.r_[-3:2]].values)
        tmp[ivar,0,-2,-2] = numba_nanmean(data[ivar,0:3,np.r_[-4:0],np.r_[-4:1]].values)
        tmp[ivar,0,-2,-1] = numba_nanmean(data[ivar,0:3,np.r_[-4:0],np.r_[-3:2]].values)
        tmp[ivar,0,-1,-2] = numba_nanmean(data[ivar,0:3,np.r_[-3:0],np.r_[-4:1]].values)
        tmp[ivar,0,-1,-1] = numba_nanmean(data[ivar,0:3,np.r_[-3:0],np.r_[-4:1]].values)
        # 4 corners in time=1-level
        tmp[ivar,1,0,0] = numba_nanmean(data[ivar,0:4,0:3,np.r_[-2:3]].values)
        tmp[ivar,1,0,1] = numba_nanmean(data[ivar,0:4,0:3,np.r_[-1:4]].values)
        tmp[ivar,1,1,0] = numba_nanmean(data[ivar,0:4,0:4,np.r_[-2:3]].values)
        tmp[ivar,1,1,1] = numba_nanmean(data[ivar,0:4,0:4,np.r_[-1:4]].values)
        tmp[ivar,1,-2,0] = numba_nanmean(data[ivar,0:4,np.r_[-4:0],np.r_[-2:3]].values)
        tmp[ivar,1,-2,1] = numba_nanmean(data[ivar,0:4,np.r_[-4:0],np.r_[-1:4]].values)
        tmp[ivar,1,-1,0] = numba_nanmean(data[ivar,0:4,np.r_[-3:0],np.r_[-2:3]].values)
        tmp[ivar,1,-1,1] = numba_nanmean(data[ivar,0:4,np.r_[-3:0],np.r_[-1:4]].values)
        tmp[ivar,1,0,-2] = numba_nanmean(data[ivar,0:4,0:3,np.r_[-4:1]].values)
        tmp[ivar,1,0,-1] = numba_nanmean(data[ivar,0:4,0:3,np.r_[-3:2]].values)
        tmp[ivar,1,1,-2] = numba_nanmean(data[ivar,0:4,0:4,np.r_[-4:1]].values)
        tmp[ivar,1,1,-1] = numba_nanmean(data[ivar,0:4,0:4,np.r_[-3:2]].values)
        tmp[ivar,1,-2,-2] = numba_nanmean(data[ivar,0:4,np.r_[-4:0],np.r_[-4:1]].values)
        tmp[ivar,1,-2,-1] = numba_nanmean(data[ivar,0:4,np.r_[-4:0],np.r_[-3:2]].values)
        tmp[ivar,1,-1,-2] = numba_nanmean(data[ivar,0:4,np.r_[-3:0],np.r_[-4:1]].values)
        tmp[ivar,1,-1,-1] = numba_nanmean(data[ivar,0:4,np.r_[-3:0],np.r_[-4:1]].values)
        # 4 corners in time=-1-level
        tmp[ivar,-1,0,0] = numba_nanmean(data[ivar,np.r_[-3:1],0:3,np.r_[-2:3]].values)
        tmp[ivar,-1,0,1] = numba_nanmean(data[ivar,np.r_[-3:1],0:3,np.r_[-1:4]].values)
        tmp[ivar,-1,1,0] = numba_nanmean(data[ivar,np.r_[-3:1],0:4,np.r_[-2:3]].values)
        tmp[ivar,-1,1,1] = numba_nanmean(data[ivar,np.r_[-3:1],0:4,np.r_[-1:4]].values)
        tmp[ivar,-1,-2,0] = numba_nanmean(data[ivar,np.r_[-3:1],np.r_[-4:0],np.r_[-2:3]].values)
        tmp[ivar,-1,-2,1] = numba_nanmean(data[ivar,np.r_[-3:1],np.r_[-4:0],np.r_[-1:4]].values)
        tmp[ivar,-1,-1,0] = numba_nanmean(data[ivar,np.r_[-3:1],np.r_[-3:0],np.r_[-2:3]].values)
        tmp[ivar,-1,-1,1] = numba_nanmean(data[ivar,np.r_[-3:1],np.r_[-3:0],np.r_[-1:4]].values)
        tmp[ivar,-1,0,-2] = numba_nanmean(data[ivar,np.r_[-3:1],0:3,np.r_[-4:1]].values)
        tmp[ivar,-1,0,-1] = numba_nanmean(data[ivar,np.r_[-3:1],0:3,np.r_[-3:2]].values)
        tmp[ivar,-1,1,-2] = numba_nanmean(data[ivar,np.r_[-3:1],0:4,np.r_[-4:1]].values)
        tmp[ivar,-1,1,-1] = numba_nanmean(data[ivar,np.r_[-3:1],0:4,np.r_[-3:2]].values)
        tmp[ivar,-1,-2,-2] = numba_nanmean(data[ivar,np.r_[-3:1],np.r_[-4:0],np.r_[-4:1]].values)
        tmp[ivar,-1,-2,-1] = numba_nanmean(data[ivar,np.r_[-3:1],np.r_[-4:0],np.r_[-3:2]].values)
        tmp[ivar,-1,-1,-2] = numba_nanmean(data[ivar,np.r_[-3:1],np.r_[-3:0],np.r_[-4:1]].values)
        tmp[ivar,-1,-1,-1] = numba_nanmean(data[ivar,np.r_[-3:1],np.r_[-3:0],np.r_[-4:1]].values)
        # 4 corners in time=-2-level
        tmp[ivar,-2,0,0] = numba_nanmean(data[ivar,np.r_[-4:1],0:3,np.r_[-2:3]].values)
        tmp[ivar,-2,0,1] = numba_nanmean(data[ivar,np.r_[-4:1],0:3,np.r_[-1:4]].values)
        tmp[ivar,-2,1,0] = numba_nanmean(data[ivar,np.r_[-4:1],0:4,np.r_[-2:3]].values)
        tmp[ivar,-2,1,1] = numba_nanmean(data[ivar,np.r_[-4:1],0:4,np.r_[-1:4]].values)
        tmp[ivar,-2,-2,0] = numba_nanmean(data[ivar,np.r_[-4:1],np.r_[-4:0],np.r_[-2:3]].values)
        tmp[ivar,-2,-2,1] = numba_nanmean(data[ivar,np.r_[-4:1],np.r_[-4:0],np.r_[-1:4]].values)
        tmp[ivar,-2,-1,0] = numba_nanmean(data[ivar,np.r_[-4:1],np.r_[-3:0],np.r_[-2:3]].values)
        tmp[ivar,-2,-1,1] = numba_nanmean(data[ivar,np.r_[-4:1],np.r_[-3:0],np.r_[-1:4]].values)
        tmp[ivar,-2,0,-2] = numba_nanmean(data[ivar,np.r_[-4:1],0:3,np.r_[-4:1]].values)
        tmp[ivar,-2,0,-1] = numba_nanmean(data[ivar,np.r_[-4:1],0:3,np.r_[-3:2]].values)
        tmp[ivar,-2,1,-2] = numba_nanmean(data[ivar,np.r_[-4:1],0:4,np.r_[-4:1]].values)
        tmp[ivar,-2,1,-1] = numba_nanmean(data[ivar,np.r_[-4:1],0:4,np.r_[-3:2]].values)
        tmp[ivar,-2,-2,-2] = numba_nanmean(data[ivar,np.r_[-4:1],np.r_[-4:0],np.r_[-4:1]].values)
        tmp[ivar,-2,-2,-1] = numba_nanmean(data[ivar,np.r_[-4:1],np.r_[-4:0],np.r_[-3:2]].values)
        tmp[ivar,-2,-1,-2] = numba_nanmean(data[ivar,np.r_[-4:1],np.r_[-3:0],np.r_[-4:1]].values)
        tmp[ivar,-2,-1,-1] = numba_nanmean(data[ivar,np.r_[-4:1],np.r_[-3:0],np.r_[-4:1]].values)
        print(ivar)
    return


filepath = '/net/so4/landclim/bverena/large_files/data_small.nc'

# create example array
print(f'read data array')
data = xr.open_dataarray(filepath)

# subset more for speedup of first tests
print(f'subset even more because very large dataset')
data = data[:,::100,:,:]
print(data.shape)

# numba njit
@numba.njit
def numba_nanmean(values):
    return np.nanmean(values)
tic = datetime.now()
footprint = np.ones((1,5,5,5))
tmp = generic_filter(data, numba_nanmean, footprint=footprint, mode='nearest')
toc = datetime.now()
print(f'numba without halo_update {toc-tic}')

# numba njit
@numba.njit
def numba_nanmean(values):
    return np.nanmean(values)
tic = datetime.now()
footprint = np.ones((1,5,5,5))
result = generic_filter(data, numba_nanmean, footprint=footprint, mode='nearest')
halo_update(data,result)
toc = datetime.now()
vali = np.isclose(result[:,2:-2,2:-2,2:-2], tmp[:,2:-2,2:-2,2:-2], equal_nan=True).all()
print(f'numba with halo_update validates outside halos against generic_filter(..., mode=nearest): {vali}')
tmp = generic_filter(data, numba_nanmean, footprint=footprint, mode='wrap')
vali = np.isclose(result[:,2:-2,2:-2,:], tmp[:,2:-2,2:-2,:], equal_nan=True).all()
print(f'numba with halo_update validates inside longitudinal halos against generic_filter(..., mode=wrap): {vali}')
print(f'numba with halo_update {toc-tic}')

"""
# unnecessary, ignore

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


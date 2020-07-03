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

import numpy as np
from mpi4py import MPI
#from partitioner import Partitioner

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(rank)

nx = 10
ny = 20
nz = 100



#shape = (3, 30, 72, 140) # real shape is (22, 3653, 720, 1440)
frac_missing = 0.42
filepath = '/net/so4/landclim/bverena/large_files/data_small.nc'

# create example array
print(f'create data array')
data_0 = xr.open_dataarray(filepath)

# subset more for speedup of first tests
print(f'subset even more because very large dataset')
data_0 = data_0[:,::1000,:,:]

shape = np.shape(data_0)

# gapfilling the missing values with spatiotemporal mean
print('gapfilling missing values with spatiotemporal mean')

tic = datetime.now()

result = np.zeros(shape)
result[:,:,:,:] = np.nan

result = np.empty((shape[1], shape[2], shape[3]))

print("Rank {} has result array of shape {}".format(rank, np.shape(result)))

#TODO: try simplifying this as simply var = rank
if rank == 0:
    var = 0
elif rank == 1:
    var = 1
elif rank == 2:
    var = 2

for t in range(2,shape[1]-2):
    print('new time t = '+str(t))
    for i in range(2,shape[2]-2):
        for j in range(2,shape[3]-2):
            tmp = 0
            k = 0
            values = data_0[var,t-2:t+2,i-2:i+1,j-2:j+2]
            #print(np.shape(data), np.shape(values))
            values[2,2,2] = np.nan # changed this to nan so that it gets ignored
            values = values.values.flatten()
            for v in values:
                if ~np.isnan(v):
                    tmp = tmp + v
                    k = k + 1
            if k != 0:
                result[t,i,j] = tmp / max(k,1)

toc = datetime.now()
print(f'this filter function took {toc-tic}')

recvbuf = None
if rank == 0:
    recvbuf = np.empty(3, np.shape(result)[0], np.shape(result)[1], np.shape(result)[2])
    print(np.shape(recvbuf))
comm.Gather(result, recvbuf, root=0)

#if rank == 0:
#    for i in range(size):
#        assert np.allclose(recvbuf[i,:], i)

if rank == 0:
    data_0 = data_0.fillna(recvbuf)
    from unittest_simple import test_simple
    res = xr.open_dataarray('baseline_result.nc')
    test_simple(recvbuf, res)

# test if results are the same as in "ground truth"
print(rank)


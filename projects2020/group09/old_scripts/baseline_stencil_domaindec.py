"""
Example for workflow for gapfilling remote sensing data from diverse sources
Explicit stencil filter

sources:
http://numba.pydata.org/numba-doc/latest/user/stencil.html
https://examples.dask.org/applications/stencils-with-numba.html

    @author: verena bessenbacher
    @date: 12 06 2020
"""

import numpy as np
from datetime import datetime
import xarray as xr
import numba
from mpi4py import MPI
from partitioner import Partitioner


@numba.stencil
def _nanmean(v, w):
    return (v[-1,0,0] * w[-1,0,0] + v[+1,0,0] * w[+1,0,0] +
            v[0,-1,0] * w[0,-1,0] + v[0,+1,0] * w[0,+1,0] +
            v[-1,-1,0] * w[-1,-1,0] + v[+1,+1,0] * w[+1,+1,0] +
            v[-1,+1,0] * w[-1,+1,0] + v[+1,-1,0] * w[+1,-1,0] +

            v[-1,0,1] * w[-1,0,1] + v[+1,0,1] * w[+1,0,1] +
            v[0,-1,1] * w[0,-1,1] + v[0,+1,1] * w[0,+1,1] +
            v[-1,-1,1] * w[-1,-1,1] + v[+1,+1,1] * w[+1,+1,1] +
            v[-1,+1,1] * w[-1,+1,1] + v[+1,-1,1] * w[+1,-1,1] +
            v[0,0,1] * w[0,0,1] +

            v[-1,0,-1] * w[-1,0,-1] + v[+1,0,-1] * w[+1,0,-1] +
            v[0,-1,-1] * w[0,-1,-1] + v[0,+1,-1] * w[0,+1,-1] +
            v[-1,-1,-1] * w[-1,-1,-1] + v[+1,+1,-1] * w[+1,+1,-1] +
            v[-1,+1,-1] * w[-1,+1,-1] + v[+1,-1,-1] * w[+1,-1,-1] +
            v[0,0,-1] * w[0,0,-1]) / max(w[-1,0,0] + 
            
            w[+1,0,0] + w[0,-1,0] + w[0,+1,0] +
            w[-1,-1,0] + w[+1,+1,0] + w[-1,+1,0] + w[+1,-1,0] +

            w[-1,0,1] + w[+1,0,1] + w[0,-1,1] + w[0,+1,1] +
            w[-1,-1,1] + w[+1,+1,1] + w[-1,+1,1] + w[+1,-1,1] +
            w[0,0,1] +

            w[-1,0,-1] + w[+1,0,-1] + w[0,-1,-1] + w[0,+1,-1] +
            w[-1,-1,-1] + w[+1,+1,-1] + w[-1,+1,-1] + w[+1,-1,-1] +
            w[0,0,-1],1)


@numba.njit
def nanmean(v,w):
    return _nanmean(v,w)

filepath = '/net/so4/landclim/bverena/large_files/data_small.nc'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(rank)

# open data
# Every rank is reading in the file. This is probably not optimal, rather one rank should read the file and send each variable to a different rank.
data = np.empty(1)
nx = None
ny = None
nz = None
if rank == 0:
    print(f'open data')
    data_orig = xr.open_dataarray(filepath)
    # let's first try only one var
    data = data_orig[0,:,:,:].copy()
    shape = np.shape(data)
    nx = shape[0]
    ny = shape[1]
    nz = shape[2]
# making shape parameters available everywhere
nx = comm.bcast(nx, root=0)
ny = comm.bcast(ny, root=0)
nz = comm.bcast(nz, root=0)

print(nx, ny, nz)
# setting up the partitioner
# the field dimensions need to be the real ones - the halo points.
p = Partitioner(comm, [nx, ny-2*2, nz-2*2], num_halo=2)

# distribute the work onto the ranks
data_work = p.scatter(data)

"""
# subset more for speedup of first tests
print(f'subset even more because very large dataset')
data = data[:,::10,:,:]
"""

# create a mask of nans
mask = ~np.isnan(data_work) # nan values have zero weight (i.e. are False)

# gapfilling the missing values with spatiotemporal mean
print('gapfilling missing values with spatiotemporal mean')
tic = datetime.now()
result = _nanmean(data_work[:,:,:], mask[:,:,:])
#result = _nanmean(data.values[1,:,:,:], mask.values[1,:,:,:])
#result = _nanmean(data.values[2,:,:,:], mask.values[2,:,:,:])
toc = datetime.now()
print(f'this filter function took {toc-tic}')
"""
tic = datetime.now() # slightly slower
result = nanmean(data.values[0,:,:,:], mask.values[0,:,:,:])
result = nanmean(data.values[1,:,:,:], mask.values[1,:,:,:])
result = nanmean(data.values[2,:,:,:], mask.values[2,:,:,:])
toc = datetime.now()
print(f'this filter function took {toc-tic}')
"""

# wait here until each process has finished before getting the results and testing (which breaks everything when an error occurs
comm.Barrier()
data = p.gather(result)

if rank == 0:
    data_out = data_orig.fillna(data)
    # test if results are the same as in "ground truth"
    from unittest_simple import test_simple
    res = xr.open_dataarray('baseline_result.nc')
    test_simple(data_out, res) # test fails bec one dim missing

# my PhD Project goes on with:
# gapfill each variable by regressing over all the others
# in an iterative EM-like fashion 
# with spatiotemporal gapfill as initial guess
# until estimates for missing values converge

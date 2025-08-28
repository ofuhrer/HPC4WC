import sys
from stencil2d_gt4py import time_stencil
import os
import numpy as np

print(f"{sys.argv[0]} started\n")

if len(sys.argv) == 2:
    filename_csv = sys.argv[1]
else:
    print(f"Usage: {sys.argv[0]} <filename_csv>")
    sys.exit(1)

# iters, zsizes, nxy as in stencil_gpu_scaling_265510.csv
iters = 2**np.arange(5,11)
zsizes = 2**np.arange(3,14)
# zsizes = 2**np.arange(3,17) # too large for GPU memory --> crashes
nxy = 128

# less work in case of DEBUG
env_var_debug = os.environ.get("DEBUG")
if env_var_debug:
    print(f'env var "DEBUG"={env_var_debug}')
    iters = 2**np.arange(5,8)
    zsizes = 2**np.arange(3,7)
        
    # moderate work in case of DEBUG="M"
    if env_var_debug == "M": # works
        iters = 2**np.arange(8,11)
        zsizes = 2**np.arange(7,12)
    elif env_var_debug == "L": # works
        iters = 2**np.arange(10,11)
        zsizes = 2**np.arange(12,14)
    elif env_var_debug == "XL": # too large -> crashed
        iters = 2**np.arange(10,11)
        zsizes = 2**np.arange(13,15)

# make timing with or without the data transfer to and from GPU based on environment variable
env_var_notransfer = os.environ.get("NOTRANSFER")
incl_transfer = (env_var_notransfer is None)
print(f"incl_transfer: {incl_transfer}")

# warm up runs to prepare all jit functions
print("Warm up runs", end="")
time_stencil(nxy, nxy, zsizes[-1], iters[0], verbose=False)

print("Start measurements:")
results = np.empty((len(zsizes), len(iters), 5))
for i,nz in enumerate(zsizes):
    for k,it in enumerate(iters):
        results[i,k] = time_stencil(nxy, nxy, nz, it, repeats=10)


try:
    """
    # write to csv in the following form:
    Nx,Ny,Nz,NUM_ITER,Time
    128, 128, 8, 32, 0.363128
    ...
    """
    np.savetxt(filename_csv, results.reshape(-1,5), delimiter=",", header="Nx,Ny,Nz,NUM_ITER,Time", fmt=('%d', '%d', '%d', '%d', '%.6f'), comments="")
    print(f"\nResults saved to {filename_csv}\n")
except Exception as e:
    print(f"Couldn't save to {filename_csv}.\nUnexpected error: {e}")
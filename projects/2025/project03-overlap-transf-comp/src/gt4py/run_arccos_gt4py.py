import sys
from arccos_gt4py import time_arccos
import os
import numpy as np

print(f"{sys.argv[0]} started\n")

if len(sys.argv) == 2:
    filename_csv = sys.argv[1]
else:
    print(f"Usage: {sys.argv[0]} <filename_csv>")
    sys.exit(1)

# ncalls and sizes as in run-arccos_cuda.sh
ncalls = 2**np.arange(10)
sizes = 2**np.arange(3,30,2)

# less work in case of DEBUG
env_var_debug = os.environ.get("DEBUG")
if env_var_debug:
    print(f'env var "DEBUG"={env_var_debug}')
    ncalls = 2**np.arange(4)
    sizes = 2**np.arange(3,12,2)
        
    # moderate and big debug configuration:
    if env_var_debug == "M":
        ncalls = 2**np.arange(4,7)
        sizes = 2**np.arange(11,19,2)
    elif env_var_debug == "L":
        ncalls = 2**np.array([7]) # np.arange(7,10)
        sizes = 2**np.arange(19,30,2)

env_var_notransfer = os.environ.get("NOTRANSFER")
incl_transfer = (env_var_notransfer is None)
print(f"incl_transfer: {incl_transfer}")

# higher recursion limit was necessary to compile the nested arccos calls with GT4Py
sys.setrecursionlimit(5000)

# warm up runs to prepare all jit functions
print("Warm up field ops for ", end="")
for nca in ncalls:
    print(f"{nca}, ", end="")
    time_arccos(nca, 1024, repeats=10, do_print=False, incl_transfer=incl_transfer)
print(" arccos calls.\n")

print("Start measurements:")
results = np.empty((len(ncalls), len(sizes), 3))
for i,nca in enumerate(ncalls):
    for k,size in enumerate(sizes):
        results[i,k] = time_arccos(nca, size, repeats=10, incl_transfer=incl_transfer)


try:
    np.savetxt(filename_csv, results.reshape(-1,3), delimiter=",", header="Calls,Size,Time", fmt=('%d', '%d', '%.6f'), comments="")
    print(f"\nResults saved to {filename_csv}\n")
except Exception as e:
    print(f"Couldn't save to {filename_csv}.\nUnexpected error: {e}")
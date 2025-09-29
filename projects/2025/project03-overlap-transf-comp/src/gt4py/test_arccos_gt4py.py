import sys
from arccos_gt4py import test_arccos
import os
import numpy as np

###
# Test arccos correctness for all provided numbers of arccos calls
###

print(f"{sys.argv[0]} started\n")

sys.setrecursionlimit(5000)

ncalls = 2**np.arange(10)  # as in run-arccos_cuda.sh
size = 8000  # small test array

all_close = True
for n in ncalls:
    try:
        test_arccos(n, size)

    except Exception as e:
        print(e)
        all_close = False

if all_close:
    print("\nTest successful!\n")
else:
    print("\nTest failed!\n")
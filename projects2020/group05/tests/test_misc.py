import pytest
import numpy as np
import sys
sys.path.append("..")
from shalconv.serialization import read_data, compare_data, OUT_VARS
from shalconv.funcphys import *

def test_read_inputvsinput():
    for i in range(6):
        data = read_data(i,"in")
        compare_data(data, data)

@pytest.mark.xfail
def test_read_inputvsoutput(): #should FAIL!
    for i in range(6):
        in_data = read_data(i,"in")
        out_data = read_data(i, "out")
        in_data_filtered = {k:in_data[k] for k in OUT_VARS}
        compare_data(in_data_filtered, out_data)
        
def test_fpvs_lookupvsexact():
    # Initialize look-up table
    gpvs()
    
    # Compute fpvs values
    n = 100
    rand_temp = np.linspace(180.0, 330.0, n)
    fpvs_arr = np.empty(n)
    fpvsx_arr = np.empty(n)
    
    for i in range(0, n):
        t = rand_temp[i]
        fpvs_arr[i] = fpvs(t)
        fpvsx_arr[i] = fpvsx(t)
        print(fpvs_arr[i], fpvsx_arr[i])
    
    # Validate
    if np.allclose(fpvs_arr, fpvsx_arr):
        print("Values are correct according to accuracy loss expected from interpolation!")
    else:
        print("Not all values are validated!")
        
    

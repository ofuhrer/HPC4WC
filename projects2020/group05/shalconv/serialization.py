import numpy as np
import random
import gt4py as gt
from . import BACKEND, DTYPE_FLOAT, DTYPE_INT, SERIALBOX_DIR, DATAPATH
from copy import deepcopy

import sys
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

int_vars = ["im", "ix", "km", "itc", "ntc", "ntk", "ntr", "ncloud"]
int_arrs = ['islimsk', 'kcnv', 'kbot', 'ktop', 'kpbl', 'kb', 'kbcon',
            'kbcon1', 'ktcon', 'ktcon1', 'ktconn', 'kbm', 'kmax',
            'cnvflg', 'flg']
scalar_vars = ["im", "ix", "km", "itc", "ntc", "ntk", "ntr", "ncloud",
               "clam", "c0s", "c1", "pgcon", "asolfac", "delt"]

IN_VARS = ["im", "ix", "km", "itc", "ntc", "ntk", "ntr", "ncloud",
           "clam", "c0s", "c1", "asolfac", "pgcon", "delt", "islimsk",
           "psp", "delp", "prslp", "garea", "hpbl", "dot",
           "phil", #"fscav", (not used)
           "kcnv", "kbot", "ktop", "qtr", "q1", "t1", "u1",
           "v1", "rn", "cnvw", "cnvc", "ud_mf", "dt_mf"]
OUT_VARS = ["kcnv", "kbot", "ktop", "qtr", "q1", "t1", "u1",
            "v1", "rn", "cnvw", "cnvc", "ud_mf", "dt_mf"]

def clean_numpy_dict(data_dict):
    """
    Transform array with length 1 into scalar values in data dict
    """
    for var in data_dict:
        if var in int_vars:
            data_dict[var] = DTYPE_INT(data_dict[var][0])
        elif data_dict[var].size <= 1:
            data_dict[var] = DTYPE_FLOAT(data_dict[var][0])

def data_dict_from_var_list(var_list, serializer, savepoint):
    """
    Read variables from specified savepoint in specified serializer
    """
    d = {}
    for var in var_list:
        d[var] = serializer.read(var, savepoint)
    clean_numpy_dict(d)
    return d
    
def numpy_dict_to_gt4py_dict(data_dict, backend = BACKEND):
    """
    Transform dict of numpy arrays into dict of gt4py storages, return new dict
    1d array of shape (nx) will be transformed into storage of shape (1, nx, nz)
    2d array of shape (nx, nz) will be transformed into storage of shape (1, nx, nz)
    3d array is kept the same (numpy arrays), doing slices later
    0d array will be transformed into a scalar
    """
    ix = int(data_dict["ix"])#im <= ix
    km = int(data_dict["km"])
    new_data_dict = {}
    
    for var in data_dict:
        
        data = data_dict[var]
        ndim = len(data.shape)
        #if var == "fscav":
        #    data_dict["fscav"] = data # shape = (number of tracers)
        
        if (ndim > 0) and (ndim <= 2) and (data.size >= 2):
            
            default_origin = (0, 0, 0)
            arrdata = np.zeros((1,ix,km))
            
            if ndim == 1: # 1D array (horizontal dimension)
                arrdata[...] = data[np.newaxis, :, np.newaxis]
            elif ndim == 2: #2D array (horizontal dimension, vertical dimension)
                arrdata[...] = data[np.newaxis, :, :]
                
            dtype = DTYPE_INT if var in int_arrs else DTYPE_FLOAT
            new_data_dict[var] = gt.storage.from_array(arrdata, backend, default_origin, dtype = dtype)
            
        elif ndim == 3: #3D array qntr(horizontal dimension, vertical dimension, number of tracers)
            new_data_dict[var] = deepcopy(data)
        else: # scalars
            new_data_dict[var] = deepcopy(data)
            
    return new_data_dict
    
def compare_data(exp_data, ref_data):
    """
    Compare two dicts of numpy arrays, raise error if one array in `exp_data` does not match the one in `ref_data`
    """
    wrong = []
    flag  = True
    
    for key in exp_data:
        mask = ~np.isnan(ref_data[key])
        
        if not np.allclose(exp_data[key][mask], ref_data[key][mask]):
            wrong.append(key)
            flag = False
        else:
            print(f"Successfully validate {key}!")
            
    assert flag, f"Data from exp and ref does not match for field {wrong}"

def read_data(tile, is_in, path = DATAPATH, ser_count = 0):
    """
    Read serialbox2 format data under `./data` folder with prefix of `Generator_rank{tile}`
    :param tile: specify the number of tile in data
    :type tile: int
    :param is_in: true means in, false means out
    :type is_in: boolean
    """
    #TODO: read_async and readbuffer
    serializer = ser.Serializer(ser.OpenModeKind.Read, path, "Generator_rank" + str(tile))
    inoutstr   = "in" if is_in else "out"
    sp         = ser.Savepoint(f"samfshalcnv-{inoutstr}-{ser_count:0>6d}")
    vars       = IN_VARS if is_in else OUT_VARS
    data       = data_dict_from_var_list(vars, serializer, sp)
    
    return data

def read_input_x_index(tile, ser_count, indices, path = DATAPATH):
    
    serializer = ser.Serializer(ser.OpenModeKind.Read, path, "Generator_rank" + str(tile))
    sp         = ser.Savepoint(f"samfshalcnv-in-{ser_count:0>6d}")
    vars       = set(IN_VARS) - set(scalar_vars)
    data       = data_dict_from_var_list(vars, serializer, sp)
    
    for key in data:
        
        ndim = len(data[key].shape)
        arr  = data[key]
        
        if ndim == 1 and key != "fscav":
            data[key] = arr[indices]
        elif ndim == 2:
            data[key] = arr[indices,:]
        elif ndim == 3:
            data[key] = arr[indices,:,:]
            
    return data

def read_random_input(length, ix, ntile, ncount, path = DATAPATH):
    """
    Generate input data of specified x dimension by randomly selecting columns from serialized data
    ignore `fscav` because it doesn't have x dimension
    :param length: number of columns to generate
    :param ix: original x dimension in the 1 tile and 1 savepoint of serialized data
    :param ntile: number of tiles in serialized data
    :param ncount: number of savepoints in serialized data
    :param path: path to serialized data
    """
    tile      = np.ndarray((length,), dtype=DTYPE_INT)
    ser_count = np.ndarray((length,), dtype=DTYPE_INT)
    index     = np.ndarray((length,), dtype=DTYPE_INT)
    
    for n in range(length):
        tile[n]      = random.randint(0, ntile - 1)
        ser_count[n] = random.randint(0, ncount - 1)
        index[n]     = random.randint(0, ix - 1)
        
    ind       = np.lexsort((index, ser_count, tile))
    index     = index[ind]
    ser_count = ser_count[ind]
    tile      = tile[ind]
    breaks    = []
    prev      = (tile[0], ser_count[0])
    for n in range(1, length):
        curr = (tile[n], ser_count[n])
        
        if prev != curr:
            breaks.append(n)
        
        prev = curr
        
    index     = np.split(index, breaks)
    data_list = []
    
    for i in range(len(breaks)):
        tile_i      = tile[breaks[i]]
        ser_count_i = ser_count[breaks[i]]
        index_i     = index[i]
        
        data_list.append(read_input_x_index(tile_i, ser_count_i, index_i, path=path))
    
    output = {}
    for key in data_list[0]:
        data = data_list[0][key]
        ndim = len(data.shape)
        
        if key == "fscav":
            output[key] = data
        elif ndim == 1:
            output[key] = np.zeros((length,), dtype=data.dtype, order='F')
        elif ndim == 2:
            output[key] = np.zeros((length, data.shape[1]), dtype=data.dtype, order='F')
        elif ndim == 3:
            output[key] = np.zeros((length, data.shape[1], data.shape[2]), dtype=data.dtype, order='F')
    
    breaks.append(length)
    breaks.insert(0, 0)
    
    for key in output:
        if key != "fscav":
            for i in range(len(data_list)):
                output[key][breaks[i]:breaks[i+1],...] = data_list[i][key]
    
    return output

def view_gt4pystorage(data_dict):
    """
    Cast dict of gt4py storage into dict of numpy arrays
    """
    new_data_dict = {}
    for key in data_dict:
        data = data_dict[key]
        if not isinstance(data, np.ndarray): data.synchronize()
        new_data_dict[key] = data.view(np.ndarray)
        
    return new_data_dict

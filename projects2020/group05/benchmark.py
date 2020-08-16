from shalconv.serialization import read_random_input, read_data, numpy_dict_to_gt4py_dict, scalar_vars
from shalconv.samfshalcnv import samfshalcnv_func
from shalconv import DTYPE_INT, BACKEND, ISDOCKER
from time import time
import numpy as np
import numpy.f2py, os


def carray2fortranarray(data_dict):
    
    for key in data_dict:
        if isinstance(data_dict[key], np.ndarray):
            data_dict[key] = np.asfortranarray(data_dict[key])
            
    return data_dict


def samfshalcnv_fort(data_dict):
    
    im      = data_dict["im"]
    ix      = data_dict["ix"]
    km      = data_dict["km"]
    delt    = data_dict["delt"]
    itc     = data_dict["itc"]
    ntc     = data_dict["ntc"]
    ntk     = data_dict["ntk"]
    ntr     = data_dict["ntr"]
    delp    = data_dict["delp"]
    prslp   = data_dict["prslp"]
    psp     = data_dict["psp"]
    phil    = data_dict["phil"]
    qtr     = data_dict["qtr"][:,:,:ntr+2]
    q1      = data_dict["q1"]
    t1      = data_dict["t1"]
    u1      = data_dict["u1"]
    v1      = data_dict["v1"]
    rn      = data_dict["rn"]
    kbot    = data_dict["kbot"]
    ktop    = data_dict["ktop"]
    kcnv    = data_dict["kcnv"]
    islimsk = data_dict["islimsk"]
    garea   = data_dict["garea"]
    dot     = data_dict["dot"]
    ncloud  = data_dict["ncloud"]
    hpbl    = data_dict["hpbl"]
    ud_mf   = data_dict["ud_mf"]
    dt_mf   = data_dict["dt_mf"]
    cnvw    = data_dict["cnvw"]
    cnvc    = data_dict["cnvc"]
    clam    = data_dict["clam"]
    c0s     = data_dict["c0s"]
    c1      = data_dict["c1"]
    pgcon   = data_dict["pgcon"]
    asolfac = data_dict["asolfac"]
    
    import shalconv_fortran
    shalconv_fortran.samfshalconv_benchmark.samfshalcnv(
                 im = im, ix = ix, km = km, delt = delt, itc = itc,
                 ntc = ntc, ntk = ntk, ntr = ntr, delp = delp,
                 prslp = prslp, psp = psp, phil = phil, qtr = qtr,
                 q1 = q1, t1 = t1, u1 = u1, v1 = v1,
                 rn = rn, kbot = kbot, ktop = ktop, kcnv = kcnv,
                 islimsk = islimsk, garea = garea, dot = dot,
                 ncloud = ncloud, hpbl = hpbl, ud_mf = ud_mf,
                 dt_mf = dt_mf, cnvw = cnvw, cnvc = cnvc, clam = clam,
                 c0s = c0s, c1 = c1, pgcon = pgcon, asolfac = asolfac )


def run_model(ncolumns, nrun = 10, compile_gt4py = True):
    
    ser_count_max = 19
    num_tiles = 6
    
    input_0 = read_data(0, True)
    
    ix = input_0["ix"]
    length = DTYPE_INT(ncolumns)
    
    times_gt4py = np.zeros(nrun)
    times_fortran = np.zeros(nrun)
    
    for i in range(nrun):
        
        data = read_random_input(length, ix, num_tiles, ser_count_max)
        
        for key in scalar_vars:
            data[key] = input_0[key]
            
        data["ix"] = length
        data["im"] = length
        data_gt4py = numpy_dict_to_gt4py_dict(data)
        data_fortran = carray2fortranarray(data)
        
        if i == 0 and compile_gt4py: samfshalcnv_func(data_gt4py)
        
        # Time GT4Py
        tic = time()
        samfshalcnv_func(data_gt4py)
        toc = time()
        times_gt4py[i] = toc - tic
        
        # Time Fortran
        tic = time()
        samfshalcnv_fort(data_fortran)
        toc = time()
        times_fortran[i] = toc - tic
        
    return times_gt4py, times_fortran


if __name__ == "__main__":
    
    lengths = [32, 128, 512, 2048, 8192, 32768, 131072] #524288]
    nrun = 10
    time_mat_gt4py = np.zeros((nrun, len(lengths)))
    time_mat_fortran = np.zeros((nrun, len(lengths)))

    print("Compiling fortran code")
    f2cmp = "--f2cmap tests/fortran/.f2py_f2cmap" if ISDOCKER else ""
    os.system(f"f2py {f2cmp} -c -m shalconv_fortran tests/fortran/samfshalconv_benchmark.f90")

    print(f"Benchmarking samfshalcnv with backend: {BACKEND}")
    
    for i in range(len(lengths)):
        
        length = lengths[i]
        times_gt4py, times_fortran = run_model(length, nrun, i==0)
        time_mat_gt4py[:,i] = times_gt4py
        time_mat_fortran[:,i] = times_fortran
        
        print(f"ix = {length}, Run time: Avg {times_gt4py.mean():.3f}, Std {np.std(times_gt4py):.3e} seconds")
        print(f"Fortran run time: Avg {times_fortran.mean():.3e}, Std {np.std(times_fortran):.3e} seconds")
    
    np.savetxt(f"times-gt4py-{BACKEND}.csv", time_mat_gt4py, delimiter=",")
    np.savetxt(f"times-fortran-{BACKEND}.csv", time_mat_fortran, delimiter=",")

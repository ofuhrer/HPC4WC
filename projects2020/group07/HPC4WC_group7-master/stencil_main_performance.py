# ******************************************************
#     Program: stencil_main.py
#      Author: HPC4WC Group 7
#        Date: 02.07.2020
# Description: Access different stencil functions via Commandline (click)
# ******************************************************

import time
import numpy as np
import click
import matplotlib
import sys
import math
from numba import njit, cuda
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

try: 
    import cupy as cp
except ImportError:
        cp=np

from functions import serialization

from functions import stencils_numpy
from functions import stencils_numba_vector_decorator
from functions import stencils_numba_loop
from functions import stencils_numba_stencil
from functions import stencils_numba_cuda
from functions import stencils_gt4py

from functions.timing import get_time
from functions.halo_functions import update_halo, add_halo_points, remove_halo_points



@click.command()
@click.option(
    "--nx", type=int, required=True, help="Number of gridpoints in x-direction"
)
@click.option(
    "--ny", type=int, required=True, help="Number of gridpoints in y-direction"
)
@click.option(
    "--nz", type=int, required=True, help="Number of gridpoints in z-direction"
)
@click.option(
    "--stencil_name",
    type=str,
    required=True,  
    help='Specify which stencil to use. Options are ["test", "laplacian1d", "laplacian2d","laplacian3d","FMA","lapoflap1d", "lapoflap2d", "lapoflap3d","test_gt4py"]',
)
@click.option(
    "--backend",
    type=str,
    required=True,
    help='Options are ["numpy", "numba_vector_function", "numba_vector_decorator", numba_loop", "numba_stencil", "numbavectorize", "gt4py", "cupy"]',
)
@click.option(
    "--num_iter", type=int, default=1, help="Number of iterations",
)
@click.option(
    "--df_name",
    type=str,
    default="df",
    help="Name of evaluation dataframe. A new name creates a new df, the same name adds a column to the already existing df.",
)
@click.option(
    "--save_runtime",
    type=bool,
    default=False,
    help="Save the individual runtimes into a df.",
)
@click.option(
    "--numba_parallel",
    type=bool,
    default=False,
    help="True to enable parallel execution of Numba stencils.",
)
@click.option(
    "--numba_cudadevice",
    type=bool,
    default=False,
    help="True to enable storage allocation on GPU.",
)
@click.option(
    "--gt4py_backend",
    type=str,
    default="numpy",
    help="GT4Py backend. Options are: numpy, gtx86, gtmc, gtcuda.",
)



def main(
    nx,
    ny,
    nz,
    backend,
    stencil_name,
    num_iter=1,
    df_name="df",
    save_runtime=False,
    numba_parallel=False,
    numba_cudadevice=False,
    gt4py_backend="numpy",
):
    """Performance assesment driver for high-level comparison of stencil computation in python."""

    assert 1 < nx <= 1024 * 1024, "You have to specify a reasonable value for nx"
    assert 1 < ny <= 1024 * 1024, "You have to specify a reasonable value for ny"
    assert 1 < nz <= 1024, "You have to specify a reasonable value for nz"
    assert (
        0 < num_iter <= 1024 * 1024
    ), "You have to specify a reasonable value for num_iter"

    stencil_name_list = [ 
        "test",
        "laplacian1d",
        "laplacian2d",
        "laplacian3d",
        "FMA",
        "lapoflap1d",
        "lapoflap2d",
        "lapoflap3d",    
        "test_gt4py",
    ]
    if stencil_name not in stencil_name_list:
        print(
            "please make sure you choose one of the following stencil: {}".format(
                stencil_name_list
            )
        )
        sys.exit(0)

    backend_list = [
        "numpy",
        "numba_vector_function",
        "numba_vector_decorator",
        "numba_loop",
        "numba_stencil",
        "numba_cuda",
        "gt4py",
        "cupy",
    ]
    if backend not in backend_list:
        print(
            "please make sure you choose one of the following backends: {}".format(
                backend_list
            )
        )
        sys.exit(0)
        
    gt4py_backend_list = [
        "numpy", 
        "gtx86", 
        "gtmc", 
        "gtcuda"
    ]
    if gt4py_backend not in gt4py_backend_list:
        print(
            "please make sure you choose one of the following backends: {}".format(
                gt4py_backend_list
            )
        )
        sys.exit(0)
        
    if backend == "gt4py" and gt4py_backend == "numpy" and stencil_name in ["lapoflap1d", "lapoflap2d", "lapoflap3d"]:
        print(
            "right now gt4py does not work for {} and lapoflapxd because of the removal of the temporary field".format(
                gt4py_backend
            )
        )
        sys.exit(0)
    
    # Create random infield
    in_field = np.random.rand(nx, ny, nz)
    
    
    # expand in_field to contain halo points
    # define value of num_halo
    if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):  
        num_halo = 1
    elif stencil_name in (
        "lapoflap1d",
        "lapoflap2d",
        "lapoflap3d",
        "test_gt4py",
    ):  
        num_halo = 2
    else:  # FMA and test
        num_halo = 0

    in_field = add_halo_points(in_field, num_halo)
    in_field = update_halo(in_field, num_halo)

    # create additional fields
    in_field2 = np.ones_like(in_field) * 2.1
    in_field3 = np.ones_like(in_field) * 4.2
    tmp_field = np.ones_like(in_field)
    out_field = np.zeros_like(in_field)
    
    # create threads for numba_cuda:
    if backend == "numba_cuda":
        threadsperblock = (8,8,8)
        
        blockspergrid_x = math.ceil(in_field.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(in_field.shape[1] / threadsperblock[1])
        blockspergrid_z = math.ceil(in_field.shape[2] / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

        if numba_cudadevice:
            in_field_d = cuda.to_device(in_field)
            in_field2_d = cuda.to_device(in_field2)
            in_field3_d = cuda.to_device(in_field3)
            out_field_d = cuda.to_device(out_field)
    
    # create fields for cupy
    if backend == "cupy":
        in_field = cp.array(in_field)
        tmp_field = cp.array(tmp_field)
        in_field2 = cp.array(in_field2)
        in_field3 = cp.array(in_field3)
        out_field = cp.array(out_field)
        
    # create fields for gt4py 
    if backend == "gt4py":
        origin = (num_halo, num_halo, num_halo)

        in_field = gt4py.storage.from_array(
            in_field, gt4py_backend, default_origin=origin
        )
        tmp_field = gt4py.storage.from_array(
            tmp_field, gt4py_backend, default_origin=origin
        )
        in_field2 = gt4py.storage.from_array(
            in_field2, gt4py_backend, default_origin=origin
        )
        in_field3 = gt4py.storage.from_array(
            in_field3, gt4py_backend, default_origin=origin
        )
        out_field = gt4py.storage.from_array(
            out_field, gt4py_backend, default_origin=origin
        )

    # import and possibly compile proper stencil object
    if backend == "numpy":
        stencil = eval(f"stencils_numpy.{stencil_name}")
    elif backend == "numba_vector_decorator":
        stencil = eval(f"stencils_numba_vector_decorator.{stencil_name}")
    elif backend == "numba_vector_function":
        stencil = eval(f"stencils_numpy.{stencil_name}")
        stencil = njit(stencil, parallel=numba_parallel)
    elif backend == "numba_loop":
        stencil = eval(f"stencils_numba_loop.{stencil_name}")
        stencil = njit(stencil, parallel=numba_parallel)
    elif backend == "numba_stencil":
        stencil = eval(f"stencils_numba_stencil.{stencil_name}")
        stencil = njit(stencil, parallel=numba_parallel)
    elif backend == "numba_cuda":
        stencil = eval(f"stencils_numba_cuda.{stencil_name}")
    elif backend == "gt4py": 
        stencil = eval(f"stencils_gt4py.{stencil_name}")
        stencil = gt4py.gtscript.stencil(gt4py_backend, stencil)
    else: #cupy
        stencil = eval(f"stencils_numpy.{stencil_name}")

    # warm-up caches
    if backend in (
        "numpy",
        "numba_vector_function",
        "numba_vector_decorator",
        "numba_loop",
        "numba_stencil",
        "cupy",
    ):
        if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
            stencil(in_field, out_field, num_halo)  
        elif stencil_name == "FMA":
            stencil(in_field, in_field2, in_field3, out_field, num_halo) 
        elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
            stencil(in_field, tmp_field, out_field, num_halo)  
        else:  # Test
            stencil(in_field,out_field)
    
    elif backend =="numba_cuda":
        if numba_cudadevice:
            if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
                stencil[blockspergrid, threadsperblock](in_field_d, out_field_d, num_halo)
            elif stencil_name == "FMA":
                stencil[blockspergrid, threadsperblock](in_field_d, in_field2_d, in_field3_d, out_field_d, num_halo)
            elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
                stencil(in_field_d, in_field2_d, out_field_d, num_halo,blockspergrid, threadsperblock)
            else:  # Test        
                stencil[blockspergrid, threadsperblock](in_field_d,out_field_d)
                
        else:   
            if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
                stencil[blockspergrid, threadsperblock](in_field, out_field, num_halo)
            elif stencil_name == "FMA":
                stencil[blockspergrid, threadsperblock](in_field, in_field2, in_field3, out_field, num_halo)
            elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
                stencil(in_field, in_field2, out_field, num_halo,blockspergrid, threadsperblock)
            else:  # Test        
                stencil[blockspergrid, threadsperblock](in_field,out_field)

    else:  # gt4py
        if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d", "test_gt4py"):
            stencil(
                in_field, out_field, origin=origin, domain=(nx, ny, nz),
            )
        elif stencil_name == "FMA":
            stencil(
                in_field,
                in_field2,
                in_field3,
                out_field,
                origin=origin,
                domain=(nx, ny, nz),
            )
        elif stencil_name in ("lapoflap1", "lapoflap2d",):
            stencil(
                in_field, out_field, origin=origin, domain=(nx, ny, nz),
            )
        elif stencil_name == "lapoflap3d":
            stencil(
                in_field, tmp_field, out_field, origin = origin, domain=(nx, ny, nz))
    #     #else: test

    # ----
    # time the actual work
    # Call the stencil chosen in stencil_name
    time_list = []
    num_iter +=1
    for i in range(num_iter):
        
        #update_halo( in_field, num_halo )

        if backend in (
            "numpy",
            "numba_vector_function",
            "numba_vector_decorator",
            "numba_loop",
            "numba_stencil"
        ):  
            update_halo( in_field, num_halo )
            if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
                tic = get_time()
                stencil(in_field, out_field, num_halo=num_halo)  
                toc = get_time()
            elif stencil_name == "FMA":
                tic = get_time()
                stencil(
                    in_field, in_field2, in_field3, out_field, num_halo=num_halo
                )  
                toc = get_time()
            elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
                tic = get_time()
                stencil(
                    in_field, tmp_field, out_field, num_halo=2
                )  
                toc = get_time()
            else:  # Test
                tic = get_time()
                stencil(in_field,out_field)
                toc = get_time()
        
        elif backend =="numba_cuda":
            #update_halo( in_field, num_halo )
            if numba_cudadevice:
                if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
                    tic = get_time()
                    stencil[blockspergrid, threadsperblock](in_field_d, out_field_d, num_halo)
                    toc = get_time()
                elif stencil_name == "FMA":
                    tic = get_time()
                    stencil[blockspergrid, threadsperblock](in_field_d, in_field2_d, in_field3_d, out_field_d, num_halo)
                    toc = get_time()
                elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
                    tic = get_time()
                    stencil(in_field_d, in_field2_d, out_field_d, num_halo,blockspergrid, threadsperblock)
                    toc = get_time()
                else:  # Test        
                    tic = get_time()
                    stencil[blockspergrid, threadsperblock](in_field_d,out_field_d)
                    toc = get_time()
            
            else:
                if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
                    tic = get_time()
                    stencil[blockspergrid, threadsperblock](in_field, out_field, num_halo)
                    toc = get_time()
                elif stencil_name == "FMA":
                    tic = get_time()
                    stencil[blockspergrid, threadsperblock](in_field, in_field2, in_field3, out_field, num_halo)
                    toc = get_time()
                elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
                    tic = get_time()
                    stencil(in_field, in_field2, out_field, num_halo,blockspergrid, threadsperblock)
                    toc = get_time()
                else:  # Test        
                    tic = get_time()
                    stencil[blockspergrid, threadsperblock](in_field,out_field)
                    toc = get_time()
        elif backend == "cupy":
            if stencil_name in ("laplacian1d", "laplacian2d", "laplacian3d"):
                tic = get_time()
                stencil(in_field, out_field, num_halo=num_halo)  
                toc = get_time()
            elif stencil_name == "FMA":
                tic = get_time()
                stencil(
                    in_field, in_field2, in_field3, out_field, num_halo=num_halo
                )  
                toc = get_time()
            elif stencil_name in ("lapoflap1d", "lapoflap2d", "lapoflap3d"):
                tic = get_time()
                stencil(
                    in_field, tmp_field, out_field, num_halo=2
                ) 
                toc = get_time()
            else:  # Test
                tic = get_time()
                stencil(in_field,out_field)
                toc = get_time()
            
        else:  # gt4py  
            if stencil_name in (
                "laplacian1d",
                "laplacian2d",
                "test_gt4py",
            ):
                tic = get_time()
                stencil(
                    in_field, out_field, origin=origin, domain=(nx, ny, nz),
                )
                toc = get_time()
            elif stencil_name == "laplacian3d": 
                tic = get_time()
                stencil(
                    in_field, 
                    out_field, 
                    origin = (num_halo, num_halo, num_halo-1),
                    domain = (nx, ny, nz+2)
                )
                toc = get_time()
            elif stencil_name == "FMA":
                tic = get_time()
                stencil(
                    in_field,
                    in_field2,
                    in_field3,
                    out_field,
                    origin=origin,
                    domain=(nx, ny, nz),
                )
                toc = get_time()
            elif stencil_name in ("lapoflap1d", "lapoflap2d",):
                tic = get_time()
                stencil(
                    in_field, out_field, origin=origin, domain=(nx, ny, nz),
                )
                toc = get_time()
            elif stencil_name == "lapoflap3d": 
                tic = get_time()
                stencil(
                    in_field,
                    tmp_field,
                    out_field,
                    origin = (num_halo, num_halo, num_halo-2), 
                    domain = (nx, ny, nz+4) 
                )
                toc = get_time()
            
                # else: test
        time_list.append(toc - tic)
        if i < num_iter - 1: #swap fields
            if numba_cudadevice:
                in_field_d, out_field_d = out_field_d, in_field_d
            
            else:
                in_field, out_field = out_field, in_field
                
    #first run is discarded             
    time_avg = np.average(time_list[1:])
    time_stdev = np.std(time_list[1:])
    time_total = sum(time_list[1:])
    num_iter = num_iter -1
    
    print(
        "Total worktime: {} s. In {} iteration(s) the average lapsed time for one run is {} +/- {} s".format(
            time_total, num_iter, time_avg, time_stdev
        )
    )

    if num_iter >= 20:
        time_avg_first_10 = sum(time_list[1:11]) / 10
        time_avg_last_10 = sum(time_list[-11:-1]) / 10
        print(
            "The average elapsed time of the first 10 run is {} and of the last 10 values is {}".format(
                time_avg_first_10, time_avg_last_10
            )
        )
    else:
        time_avg_first_10 = np.nan
        time_avg_last_10 = np.nan

    # Save into df for further processing
    # Save runtimes
    if (save_runtime==True):
        serialization.save_runtime_as_df(time_list[1:])
        print("Individual runtime saved in dataframe.")

    # Append row with calculated work to df
    serialization.add_data(
        df_name,
        stencil_name,
        backend,
        numba_parallel,
        numba_cudadevice,
        gt4py_backend,
        nx,
        ny,
        nz,
        num_iter,
        time_total,
        time_avg,
        time_stdev,
        time_avg_first_10,
        time_avg_last_10,
    )


if __name__ == "__main__":
    main()

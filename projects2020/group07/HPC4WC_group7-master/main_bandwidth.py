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

from functions.timing import get_time

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
    "--backend",
    type=str,
    required=True,
    help='Options are ["cupy", "numba_cuda", "gt4py"]',
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

def main(
    nx,
    ny,
    nz,
    backend,
    num_iter=1,
    df_name="df",
    save_runtime=False
):
    """Driver to measure the bandwidth used by different backends."""

    assert 1 < nx <= 1024 * 1024, "You have to specify a reasonable value for nx"
    assert 1 < ny <= 1024 * 1024, "You have to specify a reasonable value for ny"
    assert 1 < nz <= 1024, "You have to specify a reasonable value for nz"
    assert (
        0 < num_iter <= 1024 * 1024
    ), "You have to specify a reasonable value for num_iter"

    backend_list = [
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

    # Create random infield
    in_field = np.random.rand(nx, ny, nz)
    
    #create field on GPU device and field for copying back to host.
    if backend== "numba_cuda":
        in_field_d = cuda.device_array(shape=in_field.shape, dtype=in_field.dtype)
        in_field_h = np.empty(shape=in_field.shape, dtype=in_field.dtype)
        
    if backend == "cupy":
        in_field_d = cp.empty(shape=in_field.shape, dtype=in_field.dtype)
        in_field_h = np.empty(shape=in_field.shape, dtype=in_field.dtype)
        
    if backend == "gt4py":
        in_field_d = gt4py.storage.from_array(
                in_field, backend="gtcuda", default_origin=(0,0,0))
    
    
    # time the data transefer
    time_list = []
    time_list2 = []
    for i in range(num_iter):
        #create fields for numba_cuda
        if backend == "numba_cuda":
            # create threads for numba_cuda:
            threadsperblock = (8,8,8)

            blockspergrid_x = math.ceil(in_field.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(in_field.shape[1] / threadsperblock[1])
            blockspergrid_z = math.ceil(in_field.shape[2] / threadsperblock[2])
            blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

            #numba_cudadevice:
            tic = get_time()
            cuda.to_device(in_field, to = in_field_d)
            toc = get_time() 
            
            tic2=get_time()
            in_field_d.copy_to_host(ary = in_field_h)
            toc2=get_time()

        # create fields for cupy
        if backend == "cupy":
            tic = get_time()
            in_field_d.set(arr = in_field)
            toc = get_time()                             
            
            tic2=get_time()
            in_field_d.get(out = in_field_h)
            toc2=get_time()
            
        # create fields for gt4py 
        if backend == "gt4py":
            in_field_d.synchronize()
            tic=get_time()
            in_field_d.host_to_device(force=True)
            toc=get_time()
            
            in_field_d.synchronize()
            tic2=get_time()
            in_field_d.device_to_host(force=True)
            toc2=get_time()
            
        time_cpu_to_gpu = toc -tic
        time_gpu_to_cpu = toc2 - tic2

        time_list.append(toc - tic)
        time_list2.append(toc2 - tic2)

    time_avg_cpu_to_gpu = np.average(time_list[:])
    time_stdev_cpu_to_gpu = np.std(time_list[:])
    time_total_cpu_to_gpu = sum(time_list[:])

    time_avg_gpu_to_cpu = np.average(time_list2[:])
    time_stdev_gpu_to_cpu = np.std(time_list2[:])
    time_total_gpu_to_cpu = sum(time_list2[:])
    
    print(
        "Total transfertime from CPU to GPU: {} s. In {} iteration(s) the average lapsed time for one transfer is {} +/- {} s".format(
            time_total_cpu_to_gpu, num_iter, time_avg_cpu_to_gpu, time_stdev_cpu_to_gpu
        )
    )
    
    print(
        "Total transfertime from GPU to CPU: {} s. In {} iteration(s) the average lapsed time for one transfer is {} +/- {} s".format(
            time_total_gpu_to_cpu, num_iter, time_avg_gpu_to_cpu, time_stdev_gpu_to_cpu
        )
    )
    
    # compute size of transferred data
    num_elements = nx * ny * nz
    number_of_bytes = 8 * num_elements
    number_of_gbytes = number_of_bytes / 1024**3
    print("data transferred = {} GB".format(number_of_gbytes))
    
    # theoretical peak memory bandwidth
    peak_bandwidth_in_gbs = 32
    print("peak memory bandwidth = {} GB/s".format(peak_bandwidth_in_gbs))

    # memory bandwidth CPU to GPU
    memory_bandwidth_in_gbs_cpu_to_gpu = number_of_gbytes/time_avg_cpu_to_gpu
    memory_bandwidth_stdev_cpu_to_gpu = number_of_gbytes/time_stdev_cpu_to_gpu
    print("memory bandwidth = {:8.5f} GB/s".format(memory_bandwidth_in_gbs_cpu_to_gpu))

    # memory bandwidth GPU to CPU
    memory_bandwidth_in_gbs_gpu_to_cpu = number_of_gbytes/time_avg_gpu_to_cpu
    memory_bandwidth_stdev_gpu_to_cpu = number_of_gbytes/time_stdev_gpu_to_cpu
    print("memory bandwidth = {:8.5f} GB/s".format(memory_bandwidth_in_gbs_gpu_to_cpu))

    # compute fraction of peak CPU to GPU
    fraction_of_peak_bandwidth_cpu_to_gpu = memory_bandwidth_in_gbs_cpu_to_gpu/peak_bandwidth_in_gbs
    fraction_of_peak_bandwidth_stdev_cpu_to_gpu= memory_bandwidth_stdev_cpu_to_gpu/peak_bandwidth_in_gbs
    print("peak = {:8.5f}".format(fraction_of_peak_bandwidth_cpu_to_gpu))

    # compute fraction of peak GPU to CPU
    fraction_of_peak_bandwidth_gpu_to_cpu = memory_bandwidth_in_gbs_gpu_to_cpu/peak_bandwidth_in_gbs
    fraction_of_peak_bandwidth_stdev_gpu_to_cpu= memory_bandwidth_stdev_gpu_to_cpu/peak_bandwidth_in_gbs
    print("peak = {:8.5f}".format(fraction_of_peak_bandwidth_gpu_to_cpu))
    
    # Save into df for further processing
    # Save runtimes
    if (save_runtime==True):
        serialization.save_runtime_as_df(time_list[:])
        print("Individual runtime saved in dataframe.")

    # Append row with calculated work to df
    serialization.add_data_bandwidth(
        df_name+"_cpu_to_gpu",
        backend,
        nx,
        ny,
        nz,
        num_iter,
        time_total_cpu_to_gpu,
        time_avg_cpu_to_gpu,
        time_stdev_cpu_to_gpu,
        number_of_gbytes,
        memory_bandwidth_in_gbs_cpu_to_gpu,
        memory_bandwidth_stdev_cpu_to_gpu,
        peak_bandwidth_in_gbs,
        fraction_of_peak_bandwidth_cpu_to_gpu,
        fraction_of_peak_bandwidth_stdev_cpu_to_gpu
    )
    serialization.add_data_bandwidth(
        df_name+"_gpu_to_cpu",
        backend,
        nx,
        ny,
        nz,
        num_iter,
        time_total_gpu_to_cpu,
        time_avg_gpu_to_cpu,
        time_stdev_gpu_to_cpu,
        number_of_gbytes,
        memory_bandwidth_in_gbs_gpu_to_cpu,
        memory_bandwidth_stdev_gpu_to_cpu,
        peak_bandwidth_in_gbs,
        fraction_of_peak_bandwidth_gpu_to_cpu,
        fraction_of_peak_bandwidth_stdev_gpu_to_cpu
    )

if __name__ == "__main__":
    main()

import argparse
import time

import numpy as np
import cupy as cp
import cupyx as cpx


def iterate_arctan(n, i):
    result = cp.ones(n, dtype=np.float64)
    for _ in range(i):
        result = cp.arctan(result)
    return result

def run_one_stream(n=2**10, niter=10**2):
    results = cpx.empty_pinned(n, dtype=np.float64)
    
    tic = time.perf_counter()
    iterate_arctan(n, niter).get(out=results)
    cp.cuda.runtime.deviceSynchronize()
    t_end = (time.perf_counter() - tic) * 1000
    
    return t_end

# to run this method, use --method A
def run_several_streams_with_size(size=3584, n=2**22, niter=20):
    if n % size:
        nstreams = n // size + 1
    else:
        nstreams = n // size
        
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(nstreams)]
    results_cpu = [cpx.empty_pinned(size, dtype=np.float64) for _ in range(nstreams)]

    tic = time.perf_counter()
    for s, res_cpu in zip(streams, results_cpu):      
        with s:
            res_gpu = iterate_arctan(size, niter)                    
            res_gpu.get(stream=s, out=res_cpu) 
            
    for s in streams:
        s.synchronize()
  
    t = (time.perf_counter() - tic) * 1000
    return t, nstreams

# to run this method, use --method B
def run_several_streams_sync(size=3584, n=2**22, niter=20):
    if n % size:
        nstreams = n // size + 1
    else:
        nstreams = n // size
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(nstreams)] 

    results_cpu = [cpx.empty_pinned(size, dtype=np.float64) for _ in range(nstreams)]
    
    # measurement
    tic = time.perf_counter()
    for s, res_cpu in zip(streams, results_cpu):      
        with s:
            res_gpu = iterate_arctan(size, niter)                   
            s.synchronize()  # wait until computation finished before going to the next iteration

            res_gpu.get(stream=s, out=res_cpu)
            
    for s in streams:
        s.synchronize()
  
    t = (time.perf_counter() - tic) * 1000
    
    return t, nstreams

# to run this method, use --method C
def run_two_streams_sync(size=3584, n=2**24, niter=20):
    if n % size:
        nstreams = n // size + 1
    else:
        nstreams = n // size
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)] 

    results_cpu = [cpx.empty_pinned(size, dtype=np.float64) for _ in range(nstreams)]
    
    # measurement
    tic = time.perf_counter()
    i=0
    for res_cpu in results_cpu:      # did check, streams are alternating
        s = streams[i%2]
        with s:    
            res_gpu = iterate_arctan(size, niter)                   
            s.synchronize()  # wait until computation finished before going to the next iteration

            res_gpu.get(stream=s, out=res_cpu)
            i += 1

    for s in streams:
        s.synchronize()
  
    t = (time.perf_counter() - tic) * 1000
    
    return t, nstreams

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program for profiling of the jupyter notebook functions")
    parser.add_argument("--method", type=str, help="Method (A = run_several_streams_with_size, B = run_several_streams_sync, C = run_two_streams_sync", required=True)
    # using type=eval allows for int and 2**xy to be accepted as input 
    parser.add_argument("--n_elems", type=eval, help="Number of elements (total size of array to be caluclated)", default=2**24)
    parser.add_argument("--block_size", type=eval, help="Block size (number of data to be computed at once per stream), int and x**y allowed", default=2**20) 
    parser.add_argument("--iter", type=int, help="Iterations of arctan (dummy computation)", default=20)
    parser.add_argument("--n_runs", type=int, help="Number of runs to average over", default = 1)
    
    args = parser.parse_args()
    
    nruns = args.n_runs
    nelems = args.n_elems
    niter = args.iter
    size = args.block_size
    
    if args.method == "A":
        # warmup
        run_one_stream(nelems, niter) # warmup

        # baseline
        t_1 = 0.
        for _ in range(nruns):
            t_1 += run_one_stream(nelems, niter)

        t_1 /= nruns

        print("=== single stream with", nelems, "elements  ===")
        print("total time:", round(t_1, 5), "ms")
        
        output = np.zeros(2)

        for _ in range(nruns):
            output += run_several_streams_with_size(size, nelems, niter)

        t_s, nstreams = output / nruns

        print("")
        print(f"=== {int(nstreams)} streams ({size} elements) ===")
        imp = (t_1 - t_s)
        print("average improvement:", round(imp, 5) , f"ms ({round(100 * imp / t_1, 1)}%)")

    elif args.method == "B":
        # warmup
        run_one_stream(nelems, niter)

        t_1 = 0.
        for _ in range(nruns):
            t_1 += run_one_stream(nelems, niter)

        t_1 /= nruns

        print("=== single stream with", nelems, "elements  ===")
        print("total time:", round(t_1, 5), "ms")

        output = np.zeros(2)

        for _ in range(nruns):
            output += run_several_streams_sync(size, nelems, niter)

        t_s, nstreams = output / nruns

        print("")
        print(f"=== {int(nstreams)} streams ({size} elements) ===")
        imp = (t_1 - t_s)
        print("average improvement:", round(imp, 5) , f"ms ({round(100 * imp / t_1, 1)}%)")
            
    elif args.method == "C":
        # warmup
        run_one_stream(nelems, niter)

        t_1 = 0.
        for _ in range(nruns):
            t_1 += run_one_stream(nelems, niter)

        t_1 /= nruns

        print("=== single stream with", nelems, "elements  ===")
        print("total time:", round(t_1, 5), "ms")

        output = np.zeros(2)

        for _ in range(nruns):
            output += run_two_streams_sync(size, nelems, niter)

        t_s, nstreams = output / nruns

        print("")
        print(f"=== {int(nstreams)} streams ({size} elements) ===")
        imp = (t_1 - t_s)
        print("average improvement:", round(imp, 5) , f"ms ({round(100 * imp / t_1, 1)}%)")
    
    # special method to show malloc behaviour, also calls run_two_streams_sync() but free's GPU memory in between runs 
    elif args.method == "malloc":
        # warmup
        run_one_stream(nelems, niter)

        t_1 = 0.
        for _ in range(nruns):
            cp.get_default_memory_pool().free_all_blocks() # free memory
            t_1 += run_one_stream(nelems, niter)

        t_1 /= nruns

        print("=== single stream with", nelems, "elements  ===")
        print("total time:", round(t_1, 5), "ms")

        output = np.zeros(2)

        for _ in range(nruns):
            cp.get_default_memory_pool().free_all_blocks() # free memory
            output += run_two_streams_sync(size, nelems, niter)

        t_s, nstreams = output / nruns

        print("")
        print(f"=== {int(nstreams)} streams ({size} elements) ===")
        imp = (t_1 - t_s)
        print("average improvement:", round(imp, 5) , f"ms ({round(100 * imp / t_1, 1)}%)")
    
    print("=== End of program ===")
from stencil2d_naive import apply_diffusion as stencil_naive
from stencil2d_numpy import apply_diffusion as stencil_numpy
from stencil2d_cupy import apply_diffusion as stencil_cupy
from stencil2d_mpi import apply_diffusion as stencil_mpi

import pandas as pd
from pprint import pprint as ppp
import numpy as np
import input_loader
import time


def main():
    problem_sizes = input_loader.load_input()
    print("Available problem sizes:", *problem_sizes, sep="\n")

    num_halo = 2

    versions = [stencil_naive, stencil_numpy, stencil_cupy]
    
    df = pd.DataFrame(columns=[f.__module__ for f in versions], dtype=np.float32)
    for i in range(len(problem_sizes)):
        df.loc[i] = [0 for _ in range(len(versions))]  # number of cols needs to match len(versions)
    ppp(df)

    # To test if all benchmarks work etc just put in problem_sizes[:1]
    # so it only runs the first (smalles) problem size
    for p, problem_size in enumerate(problem_sizes):
        print("Benching problem size {}: ".format(p), problem_size)
        (nx, ny, nz, alpha, num_iter) = problem_size

        # To test if all benchmarks work etc just put in versions[:1] so it only runs naive
        for f, func in enumerate(versions):
            print("       Version: ", func.__module__)
            if func.__module__ == "stencil2d_cupy":
                try:
                    import cupy as xp
                    print("Running on GPU with cupy (benches)")
                except ImportError:
                    xp = np
                    print("Running on CPU with numpy (benches)")
                
                in_field = xp.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo))
                in_field[
                    nz // 4 : 3 * nz // 4,
                    num_halo + ny // 4 : num_halo + 3 * ny // 4,
                    num_halo + nx // 4 : num_halo + 3 * nx // 4,
                ] = 1.0
                out_field = xp.copy(in_field)

                # warmup caches
                stencil_cupy(in_field, out_field, alpha, num_halo)

                cupy_times = []
                for _ in range(10):
                    tic = time.time()
                    stencil_cupy(in_field, out_field, alpha, num_halo, num_iter=num_iter)
                    toc = time.time()
                    cupy_times.append((toc - tic))
                df.loc[p, func.__module__] = min(cupy_times) * 1e3

            else:
                in_field = input_loader.generate_initial_array(problem_size)
                # print(np.shape(in_field))  # nz = shape[0], ny = shape[1], nx = shape[0]
                out_field = np.copy(in_field)

                # warmup caches
                stencil_naive(in_field, out_field, alpha, num_halo, num_iter)

                version_times = []
                for _ in range(10):
                    tic = time.time()
                    func(in_field, out_field, alpha, num_halo, num_iter=num_iter)
                    toc = time.time()
                    version_times.append((toc - tic))

                # multiply by 1e3 to go from seconds (units of the return value of timeit.timeit) to ms
                df.loc[p, func.__module__] = min(version_times) * 1e3

    ppp(df)
    df.to_csv("../../universal_output/py_benchmarks.csv", sep=",", index=False)


if __name__ == "__main__":
    main()

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

    versions = [stencil_mpi,]

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
            from mpi4py import MPI
            from partitioner2 import Partitioner
            import matplotlib.pyplot as plt
            from stencil2d_mpi import update_halo as update_halo_mpi

            mpi_times = []
            for _ in range(10):
                comm = MPI.COMM_WORLD

                p = Partitioner(comm, [nz, ny, nx], num_halo)

                if p.rank() == 0:
                    f = input_loader.generate_initial_array(problem_size)
                else:
                    f = None

                in_field = p.scatter(f)
                out_field = np.copy(in_field)

                f = p.gather(in_field)

                # warmup caches
                stencil_mpi(in_field, out_field, alpha, num_halo, p=p)

                comm.Barrier()

                # time the actual work
                tic = time.time()
                stencil_mpi(in_field, out_field, alpha, num_halo, num_iter=num_iter, p=p)
                toc = time.time()

                comm.Barrier()

                update_halo_mpi(out_field, num_halo, p)

                f = p.gather(out_field)

                mpi_times.append(toc - tic)

            # time.time() returns value in seconds, thus times 1e3
            df.loc[p, func.__module__] = min(mpi_times) * 1e3


    ppp(df)
    df.to_csv("../../universal_output/py_mpi_benchmarks.csv", sep=",", index=False)


if __name__ == "__main__":
    main()

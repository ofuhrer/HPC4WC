import os
import sys
from datetime import datetime
import pickle
import numpy as np

# Import functions
from stencil2d_jax_base import calculations as jax_base_calc
from stencil2d_jax import calculations as jax_calc
from stencil2d_numpy import calculations as numpy_calc
from stencil2d_torch import calculations as torch_calc
from stencil2d_numba import calculations as numba_calc

# Define paramters to test
range_nx = [16, 32, 48, 64, 80, 96, 112, 128]
range_ny = range_nx
range_nz = np.repeat(64, len(range_nx))
range_num_iter = np.repeat(128, len(range_nx))
range_precision = ["32", "64"]
functions = {"jax_base": jax_base_calc, "jax": jax_calc, "numpy": numpy_calc, "torch": torch_calc, "numba": numba_calc}

results = {}

num_reps = 5

# Delete content of folder results_tmp
folder = "results_tmp"
for filename in os.listdir(folder):
    if not filename.startswith("."):
        os.remove(os.path.join(folder, filename))


def tester():
    for nx, ny, nz, num_iter in zip(range_nx, range_ny, range_nz, range_num_iter):
        for p in range_precision:
            for n, f in functions.items():
                for r in range(num_reps):
                    results[f"{r}_nx{nx}_ny{ny}_nz{nz}_num_iter{num_iter}_p{p}_{n}"] = f(
                        nx, ny, nz, num_iter, 2, p, return_result=False, return_time=True
                    )
            with open(f"../results_tmp/{datetime.now().strftime('%Y%m%dT%H%M%S')}.pkl", "wb") as f_out:
                pickle.dump(results, f_out)


def main():
    tester()


if __name__ == "__main__":
    os.chdir(sys.path[0])  # Change the directory
    main()

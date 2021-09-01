import time
import pandas as pd
import numpy as np

from stencil import Dummy_Stencil
from numpy_laplacian import Numpy_diffusion
# from numpy_stencils import Laplacian2D, Laplacian3D, Biharmonic2D, Biharmonic3D
import numpy_stencils
import gt4py_stencils
# import jax_stencils


def time_iterations(stencil, number_of_iterations):
    # returns duration of a single stencil.run() call in ms
    stencil.activate()
    stencil.run()  # warmup

    tic = time.perf_counter()
    for i in range(number_of_iterations):
        stencil.run()
    stencil.deactivate()
    toc = time.perf_counter()
    return (toc-tic)/number_of_iterations


def main():
    number_of_itertions = 3
    ns = np.arange(100, 600, 100)
    timings_dicts  = []

    for n in ns:
        current_row = {"n":n}
        stencils = [numpy_stencils.Laplacian2D(n),
                    gt4py_stencils.Laplacian2D(n),
                    numpy_stencils.Laplacian3D(n),
                    gt4py_stencils.Laplacian3D(n),
                    numpy_stencils.Biharmonic2D(n),
                    gt4py_stencils.Biharmonic2D(n),
                    numpy_stencils.Biharmonic3D(n),
                    gt4py_stencils.Biharmonic3D(n),]
                    #jax_stencils.Laplacian2D(n)]

        for s in stencils:
            try:
                average_duration = time_iterations(s, number_of_itertions)
                print(
                    f"Stencil: {s}\nN: {n}\nNumber of Iterations: {number_of_itertions}\nTime: {average_duration}s\n")

                current_row[str(s)] = average_duration
            except:
                print(f"Error in {s}")
                current_row[str(s)] = None

        timings_dicts.append(current_row)

    timings = pd.DataFrame(timings_dicts)
    timings.to_csv("timings.csv", index=False)


if __name__ == "__main__":
    main()

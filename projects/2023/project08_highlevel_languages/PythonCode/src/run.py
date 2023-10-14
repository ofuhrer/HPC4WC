
from stencil2d_seq import apply_diffusion as stencil_naive
from stencil2d_numpy import update_halo
from stencil2d_numpy import apply_diffusion as stencil_numpy
import input_loader

import numpy as np
import matplotlib.pyplot as plt
import time

def main():
    problem_sizes = input_loader.load_input()
    problem_size = problem_sizes[0]
    num_halo = 2
    alpha = problem_size[3]
    num_iter = problem_size[4]
    plot_result = True

    in_field = input_loader.generate_initial_array(problem_size)
    # print(np.shape(in_field))  # nz = shape[0], ny = shape[1], nx = shape[0]
    out_field = np.copy(in_field)

    if plot_result:
        plt.ioff()
        plt.imshow(in_field[in_field.shape[0] // 2, :, :], origin="lower")
        plt.colorbar()
        plt.savefig("in_field.png")
        plt.close()

    out_field = np.copy(in_field)

    np.save("in_field", in_field)

    update_halo(in_field, num_halo)

    # warmup caches
    stencil_naive(in_field, out_field, alpha, num_halo)

    # time the actual work
    tic = time.time()
    stencil_naive(in_field, out_field, alpha, num_halo, num_iter=num_iter)
    toc = time.time()

    print(f"Elapsed time for work = {toc - tic} s")

    np.save("out_field", out_field)

    if plot_result:
        plt.imshow(out_field[out_field.shape[0] // 2, :, :], origin="lower")
        plt.colorbar()
        plt.savefig("out_field.png")
        plt.close()


if __name__ == "__main__":
    main()
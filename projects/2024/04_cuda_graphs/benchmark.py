import cupy as cp
import numpy as np
from baseline import stencil2d_cupy
from custom_graph_capture import stencil2d_custom_unrolled_graph
from custom_kernel import (
    stencil2d_custom_v1,
    stencil2d_custom_v2,
    stencil2d_custom_v3,
    stencil2d_custom_v4,
    stencil2d_custom_v5,
    stencil2d_custom_v6,
    stencil2d_custom_v7,
    stencil2d_custom_v8,
)
from graph_capture import (
    stencil2d_memcpy_graph,
    stencil2d_two_graphs,
    stencil2d_unrolled_graph,
)
from tqdm import tqdm

IMPLEMENTATIONS = [
    stencil2d_custom_unrolled_graph,
    stencil2d_memcpy_graph,
    stencil2d_two_graphs,
    stencil2d_unrolled_graph,
    stencil2d_cupy,
    stencil2d_custom_v1,
    stencil2d_custom_v2,
    stencil2d_custom_v3,
    stencil2d_custom_v4,
    stencil2d_custom_v5,
    stencil2d_custom_v6,
    stencil2d_custom_v7,
    stencil2d_custom_v8,
]


NUM_HALO = 2
NUM_MEASUREMENTS = 32


def main():
    sizes = [2**i for i in range(3, 11)]
    iters = [2**i + 1 for i in range(5, 11)]
    nz = 64

    times = np.zeros((len(IMPLEMENTATIONS), len(sizes), len(iters), NUM_MEASUREMENTS))

    mempool = cp.get_default_memory_pool()
    for i, implementation in enumerate(IMPLEMENTATIONS):
        # # print(implementation)
        for j, size in tqdm(
            enumerate(sizes), total=len(sizes), desc=implementation.__name__
        ):
            # # print(size)
            for k, num_iter in enumerate(iters):
                # # print(num_iter)
                for l in range(NUM_MEASUREMENTS):
                    time = implementation.main(
                        nx=size,
                        ny=size,
                        nz=nz,
                        num_iter=num_iter,
                        num_halo=NUM_HALO,
                        plot_result=False,
                        save_result=False,
                        benchmark=True,
                        return_result=False,
                    )
                    times[i, j, k, l] = time
                    mempool.free_all_blocks()

    np.save("times.npy", times)


if __name__ == "__main__":
    main()

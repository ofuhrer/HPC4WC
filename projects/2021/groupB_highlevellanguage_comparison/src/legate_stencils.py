import os
from stencil import Stencil

import legate.numpy as np
import time


def get_lap2d_time(n, number_of_iterations):
    num_halo = 2
    in_field = np.zeros((n, n))
    out_field = np.copy(in_field)

    in_field[
        num_halo + n // 4: num_halo + 3 * n // 4,
        num_halo + n // 4: num_halo + 3 * n // 4,
    ] = 1.0

    tic = time.perf_counter()
    for i in range(number_of_iterations):
        I, J = in_field.shape

        ib = num_halo
        ie = I - num_halo
        jb = num_halo
        je = J - num_halo
        
        print(f"Running stencil: {num_halo}:{ie}, {jb}:{je}")

        out_field[ib:ie, jb:je] = (
            -4.0 * in_field[ib:ie, jb:je]
            + in_field[ib - 1: ie - 1, jb:je]
            + in_field[ib + 1: ie + 1, jb:je]
            + in_field[ib:ie, jb - 1: je - 1]
            + in_field[ib:ie, jb + 1: je + 1]
        )
        print(in_field[int((ie-ib)/2),int((je-jb)/2)])
        in_field, out_field = out_field, in_field
        print(in_field[int((ie-ib)/2),int((je-jb)/2)])

    toc = time.perf_counter()
    return (toc-tic)/number_of_iterations


print(get_lap2d_time(16, 4))
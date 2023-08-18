import time
import numpy as np
from dsl.generated.main import generated_function
#from dsl.baseline.baseline_functions import mean_with_numpy, mean_without_numpy


def main(kinds=None):
    # Input fields
    nx = 10
    ny = 10
    nz = 10
    num_iter = 10
    num_halo = 2
    alpha = 1.0 / 32.0

    in_field = np.zeros((nx + 2 * num_halo, ny + 2 * num_halo,nz))

    in_field[
    num_halo + nx // 4: num_halo + 3 * nx // 4,
    num_halo + ny // 4: num_halo + 3 * ny // 4,
    nz // 4: 3 * nz // 4
    ] = 1.0

    out_field = np.copy(in_field)
    tmp_field = np.empty_like(in_field)


    # Timers
    start_times = {}
    exec_times = {}

    for kind in kinds:
        start_times[kind] = time.time()
        run_function(kind, in_field, out_field, num_halo, nx, ny, nz, num_iter,tmp_field,alpha)
        exec_times[kind] = time.time() - start_times[kind]

    # Output diagnostics
    diagnostics(exec_times)

    # TODO: Verify results


def run_function(kind, in_field, out_field, num_halo, nx, ny, nz, num_iter,tmp_field,alpha):
    if kind == "base":
        # TODO: Baseline implementation
        pass
    if kind == "gen":
        return generated_function(in_field, out_field, num_halo, nx, ny, nz, num_iter,tmp_field,alpha)


def diagnostics(times):
    max_key_length = max(len(key) for key in times.keys())

    print("================")
    print("Execution times:")
    print("================")
    for key, value in times.items():
        output = "{:<{width}}: {:>}ms".format(key, round(value * 1000, 2), width=max_key_length)
        print(output)


if __name__ == "__main__":
    kinds = ["gen", "base"]
    main(kinds=kinds)

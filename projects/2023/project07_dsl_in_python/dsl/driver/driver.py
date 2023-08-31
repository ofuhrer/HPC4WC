import time
import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
    CUDA = True
except ImportError:
    CUDA = False
    

from dsl.generated.main import generated_function
from dsl.baseline.baseline_stencil import baseline_stencil


def main(kinds=None):
    # Input fields
    nx = 20
    ny = 20
    nz = 20
    num_iter = 10
    num_halo = 2
    alpha = 1.0 / 32.0

    in_field = np.zeros((nx + 2 * num_halo, ny + 2 * num_halo, nz))

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
    out_fields = {}

    for kind in kinds:
        start_times[kind] = []
        exec_times[kind] = []
        
        for i in range(10):
            start_times[kind].append(time.time())
            out_fields[kind] = run_function(kind, in_field, out_field, num_halo, nx, ny, nz, num_iter, tmp_field, alpha)
            exec_times[kind].append(time.time() - start_times[kind][i])

    # Output validaiton
    validation(out_fields)

    print("")

    # Output diagnostics
    diagnostics(exec_times)


def run_function(kind, in_field, out_field, num_halo, nx, ny, nz, num_iter, tmp_field, alpha):
    if kind == "base":
        return baseline_stencil(in_field, out_field, num_halo, nx, ny, nz, num_iter, tmp_field, alpha)
    if kind == "generated":
        return generated_function(in_field, out_field, num_halo, nx, ny, nz, num_iter, tmp_field, alpha)
    if kind == "cuda":
        in_field_cp = cp.asarray(in_field)
        out_field_cp = cp.asarray(out_field)
        tmp_field_cp = cp.asarray(tmp_field)
        
        out_field_cp = generated_function(in_field_cp, out_field_cp, num_halo, nx, ny, nz, num_iter, tmp_field_cp, alpha)
        return out_field_cp.get()

def validation(out_fields):
    max_key_length = max(len(key) for key in out_fields.keys())

    print("================")
    print("Validaiton of output:")
    print("================")

    for key, value in out_fields.items():
        plt.imshow(out_fields[key][out_fields[key].shape[0] // 2, :, :], origin="lower")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"plot_{key}.png")
        plt.close()
        
        if key != "base":
            if np.allclose(out_fields["base"], out_fields[key], rtol=1e-5, atol=1e-8, equal_nan=True):
                output = "{:<{width}}: passed".format(key, width=max_key_length)
            else:
                output = "{:<{width}}: failed".format(key, width=max_key_length)
            print(output)


def diagnostics(times):
    max_key_length = max(len(key) for key in times.keys())

    print("================")
    print("Execution times:")
    print("================")
    for key, value in times.items():
        output = "{:<{width}}: {:>}ms, (sd: {:>}ms)".format(key, round(np.mean(value) * 1000, 2), round(np.std(value) * 1000, 2), width=max_key_length)
        print(output)


if __name__ == "__main__":
    kinds = ["generated", "base"]
    if CUDA:
        kinds.append("cuda")
    main(kinds=kinds)

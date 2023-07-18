import time
import numpy as np
from dsl.generated.main import generated_function
from dsl.baseline.baseline_functions import mean_with_numpy, mean_without_numpy


def main(kinds=None):
    # Input fields
    field1 = np.zeros([10, 10, 20])
    field2 = np.zeros([10, 10, 20])

    # Timers
    start_times = {}
    exec_times = {}

    for kind in kinds:
        start_times[kind] = time.time()
        run_function(kind, field1, field2)
        exec_times[kind] = time.time() - start_times[kind]

    # Output diagnostics
    diagnostics(exec_times)

    # TODO: Verify results


def run_function(kind, field1, field2):
    if kind == "base":
        # TODO: Baseline implementation
        return mean_without_numpy(field1)
    if kind == "gen":
        return generated_function(field1, field2)


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

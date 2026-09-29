import os
from collections.abc import Callable
from typing import cast


def available_cores() -> int:
    sched_getaffinity = getattr(os, "sched_getaffinity", None)

    if sched_getaffinity is not None:
        get_affinity = cast(Callable[[int], set[int]], sched_getaffinity)
        affinity_cores = len(get_affinity(0))
    else:
        affinity_cores = os.cpu_count() or 1

    omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_num_threads is not None:
        try:
            omp_value = int(omp_num_threads)
        except ValueError:
            return affinity_cores

        if omp_value > 0:
            return min(omp_value, affinity_cores)

    return affinity_cores

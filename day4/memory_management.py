#!/usr/bin/env python3
"""Measure explicit CuPy host/device transfer cases on one GPU.

This script is used by the Day 4 bonus exercise. It intentionally keeps the
benchmark narrow: pageable and pinned host buffers are measured for explicit
host-to-device and device-to-host copies. Managed memory is shown separately as
an allocation mode, not as another transfer-bandwidth result.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence

import cupy as cp
import numpy as np


DTYPE = np.float32


def synchronize() -> None:
    cp.cuda.Device().synchronize()


def pinned_empty(shape: tuple[int, ...], dtype: np.dtype) -> tuple[np.ndarray, object]:
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    memory = cp.cuda.alloc_pinned_memory(nbytes)
    array = np.frombuffer(memory, dtype=dtype, count=int(np.prod(shape))).reshape(shape)
    return array, memory


def best_time(action: Callable[[], object], repeat: int) -> float:
    action()
    synchronize()

    timings = []
    for _ in range(repeat):
        synchronize()
        start = time.perf_counter()
        result = action()
        synchronize()
        timings.append(time.perf_counter() - start)

        # Keep temporary arrays alive until after synchronization.
        del result

    return min(timings)


def gb_per_second(nbytes: int, elapsed: float) -> float:
    return nbytes / 1000**3 / elapsed


def benchmark_transfers(
    sizes: Sequence[int], nz: int, repeat: int
) -> list[tuple[str, float, float, float, float, float]]:
    rows = []

    for nxy in sizes:
        shape = (nz, nxy, nxy)
        nbytes = int(np.prod(shape)) * np.dtype(DTYPE).itemsize
        size_mb = nbytes / 1000**2

        pageable_in = np.full(shape, 1.0, dtype=DTYPE)
        pageable_out = np.empty_like(pageable_in)
        pinned_in, pinned_in_memory = pinned_empty(shape, DTYPE)
        pinned_out, pinned_out_memory = pinned_empty(shape, DTYPE)
        pinned_in[...] = pageable_in

        device_buffer = cp.empty(shape, dtype=cp.float32)
        device_source = cp.full(shape, 2.0, dtype=cp.float32)
        synchronize()

        pageable_h2d = best_time(lambda: device_buffer.set(pageable_in), repeat)
        pageable_d2h = best_time(lambda: device_source.get(out=pageable_out), repeat)
        pinned_h2d = best_time(lambda: device_buffer.set(pinned_in), repeat)
        pinned_d2h = best_time(lambda: device_source.get(out=pinned_out), repeat)

        rows.append(
            (
                f"{nxy}x{nxy}x{nz}",
                size_mb,
                gb_per_second(nbytes, pageable_h2d),
                gb_per_second(nbytes, pageable_d2h),
                gb_per_second(nbytes, pinned_h2d),
                gb_per_second(nbytes, pinned_d2h),
            )
        )

        del pageable_in, pageable_out, pinned_in, pinned_out
        del pinned_in_memory, pinned_out_memory
        del device_buffer, device_source
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    return rows


def managed_memory_smoke(shape: tuple[int, ...], repeat: int) -> tuple[float, float, float]:
    nbytes = int(np.prod(shape)) * np.dtype(DTYPE).itemsize
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)

    with cp.cuda.using_allocator(pool.malloc):
        managed = cp.empty(shape, dtype=cp.float32)

    elapsed = best_time(lambda: managed.fill(1.0), repeat)
    checksum = float(cp.asnumpy(managed[:1, :1, :1]).sum())
    pool.free_all_blocks()
    return nbytes / 1000**2, elapsed, checksum


def print_transfer_table(rows: list[tuple[str, float, float, float, float, float]]) -> None:
    print("\nExplicit copy bandwidths")
    print("H2D = host to device, D2H = device to host")
    print("Pinned copies use explicit pinned NumPy output/input buffers.")
    print()
    print(
        f"{'shape':>14} {'size':>12} "
        f"{'page H2D':>14} {'page D2H':>14} "
        f"{'pinned H2D':>14} {'pinned D2H':>14}"
    )
    for shape, size_mb, page_h2d, page_d2h, pinned_h2d, pinned_d2h in rows:
        print(
            shape.rjust(14),
            f"{size_mb:8.1f} MB",
            f"{page_h2d:9.1f} GB/s",
            f"{page_d2h:9.1f} GB/s",
            f"{pinned_h2d:9.1f} GB/s",
            f"{pinned_d2h:9.1f} GB/s",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--nz", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    print(f"GPU: {props['name'].decode()} (device {device.id})")
    print(f"CuPy: {cp.__version__}")

    rows = benchmark_transfers(args.sizes, args.nz, args.repeat)
    print_transfer_table(rows)

    managed_size_mb, managed_elapsed, checksum = managed_memory_smoke(
        (args.nz, args.sizes[-1], args.sizes[-1]), args.repeat
    )
    print("\nManaged-memory check")
    print(
        "Allocated CUDA managed memory and timed a GPU fill. "
        "This is not a host/device copy bandwidth measurement."
    )
    print(
        f"shape={args.sizes[-1]}x{args.sizes[-1]}x{args.nz}, "
        f"size={managed_size_mb:.1f} MB, "
        f"best fill time={managed_elapsed * 1e3:.3f} ms, "
        f"checksum={checksum:.1f}"
    )


if __name__ == "__main__":
    main()

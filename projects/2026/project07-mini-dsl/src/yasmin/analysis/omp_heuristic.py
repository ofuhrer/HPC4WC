"""Shape-based heuristics for the openMP Backend"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelConfig:
    parallelize: bool
    collapse: bool
    num_threads: int | None
    schedule_chunk: int | None


def parallel_config(
    outer_extent: int,
    inner_extent: int | None,
    *,
    max_threads: int,
    min_iters_per_thread: int,
    collapse_enabled: bool,
    schedule_kind: str | None,
) -> ParallelConfig:
    """
    Decides collapse / thread count / schedule chunk from a loop nest's actual
    trip counts (resolved from concrete shapes at compile time).

    - collapse(2) is only worth the index-decomposition overhead when the outer
      loop alone can't fill `max_threads`, but outer * inner can.
    - Requesting more threads than can be filled with >= `min_iters_per_thread`
      iterations each just adds fork/join overhead for no benefit.
    - Loops with fewer than `min_iters_per_thread` total iterations aren't worth
      parallelizing at all.
    - schedule(dynamic|guided) without a chunk defaults to 1 - an atomic
      scheduling access per iteration. If no chunk is set explicitly, one is
      picked here.
    """

    def threads_for(extent: int) -> int:
        if extent <= 0 or min_iters_per_thread <= 0:
            return 1
        return max(1, min(max_threads, extent // min_iters_per_thread))

    outer_threads = threads_for(outer_extent)

    can_collapse = collapse_enabled and inner_extent is not None and inner_extent > 0

    if can_collapse and inner_extent is not None:
        combined_extent = outer_extent * inner_extent
        combined_threads = threads_for(combined_extent)
    else:
        combined_extent = outer_extent
        combined_threads = outer_threads

    should_collapse = can_collapse and combined_threads > outer_threads

    extent = combined_extent if should_collapse else outer_extent
    num_threads = combined_threads if should_collapse else outer_threads

    if extent < min_iters_per_thread or num_threads <= 1:
        return ParallelConfig(
            parallelize=False,
            collapse=False,
            num_threads=None,
            schedule_chunk=None,
        )

    schedule_chunk = None
    if schedule_kind in ("dynamic", "guided"):
        schedule_chunk = max(1, extent // (num_threads * 4))

    return ParallelConfig(
        parallelize=True,
        collapse=should_collapse,
        num_threads=num_threads,
        schedule_chunk=schedule_chunk,
    )

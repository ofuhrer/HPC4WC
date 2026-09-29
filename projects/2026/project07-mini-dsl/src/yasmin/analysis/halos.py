from yasmin.analysis.offsets import OffsetBounds

Halo = tuple[tuple[int, int], ...]


def infer_halo(bounds: OffsetBounds) -> Halo:
    return tuple((max(0, -lower), max(0, upper)) for lower, upper in bounds)

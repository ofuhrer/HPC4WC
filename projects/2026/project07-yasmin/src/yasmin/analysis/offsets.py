from collections import defaultdict

from yasmin.analysis.accesses import collect_accesses
from yasmin.core import Field
from yasmin.ir.stencil import Expr

OffsetBounds = tuple[tuple[int, int], ...]


def infer_offset_bounds(expr: Expr) -> dict[Field, OffsetBounds]:
    accesses = collect_accesses(expr)

    bounds: dict[Field, list[list[int]]] = defaultdict(list)

    for access in accesses:
        if access.field not in bounds:
            bounds[access.field] = [[offset, offset] for offset in access.offsets]
            continue

        for dim, offsets in enumerate(access.offsets):
            bounds[access.field][dim][0] = min(bounds[access.field][dim][0], offsets)
            bounds[access.field][dim][1] = max(bounds[access.field][dim][1], offsets)

    return {
        field: tuple((lower, upper) for lower, upper in field_bounds)
        for field, field_bounds in bounds.items()
    }

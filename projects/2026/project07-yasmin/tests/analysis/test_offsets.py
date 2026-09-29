from yasmin.analysis import infer_offset_bounds
from yasmin.core import Dimension, Field, float64
from yasmin.ir.stencil import BinaryExpr, BinaryOp, FieldAccess


def test_infer_offset_bounds() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=float64)

    a = FieldAccess(u, (-2, 0))
    b = FieldAccess(u, (1, 0))
    c = FieldAccess(u, (0, 3))

    expr = BinaryExpr(
        BinaryOp.ADD,
        BinaryExpr(BinaryOp.ADD, a, b),
        c,
    )

    bounds = infer_offset_bounds(expr)

    assert bounds[u] == ((-2, 1), (0, 3))

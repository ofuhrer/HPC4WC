from yasmin.analysis import collect_accesses
from yasmin.core import Dimension, Field, float64
from yasmin.ir.stencil import BinaryExpr, BinaryOp, FieldAccess, Literal


def test_collect_accesses() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=float64)

    left = FieldAccess(u, (-1, 0))
    right = FieldAccess(u, (1, 0))

    expr = BinaryExpr(BinaryOp.ADD, left, right)

    accesses = collect_accesses(expr)

    assert accesses == (left, right)


def test_collect_accesses_from_laplacian() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=float64)

    center = FieldAccess(u, (0, 0))
    left = FieldAccess(u, (-1, 0))
    right = FieldAccess(u, (1, 0))
    up = FieldAccess(u, (0, 1))
    down = FieldAccess(u, (0, -1))

    sum_horizontal = BinaryExpr(BinaryOp.ADD, left, right)
    sum_vertical = BinaryExpr(BinaryOp.ADD, up, down)

    sum_all = BinaryExpr(BinaryOp.ADD, sum_horizontal, sum_vertical)
    four_center = BinaryExpr(BinaryOp.ADD, Literal(4.0), center)

    laplacian = BinaryExpr(BinaryOp.SUB, sum_all, four_center)

    accesses = collect_accesses(laplacian)

    assert accesses == (left, right, up, down, center)

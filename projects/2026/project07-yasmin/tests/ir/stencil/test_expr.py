from yasmin.core import Dimension, Field, float64
from yasmin.ir.stencil import (
    Assign,
    BinaryExpr,
    BinaryOp,
    FieldAccess,
    Literal,
    Operator,
)


def test_five_point_laplacian_ir() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", (x, y), dtype=float64)
    out = Field("out", (x, y), dtype=float64)

    center = FieldAccess(u, (0, 0))
    left = FieldAccess(u, (-1, 0))
    right = FieldAccess(u, (1, 0))
    up = FieldAccess(u, (0, 1))
    down = FieldAccess(u, (0, -1))

    sum_horizontal = BinaryExpr(
        BinaryOp.ADD,
        left,
        right,
    )

    sum_vertical = BinaryExpr(
        BinaryOp.ADD,
        up,
        down,
    )

    sum_all = BinaryExpr(
        BinaryOp.ADD,
        sum_horizontal,
        sum_vertical,
    )

    four_center = BinaryExpr(
        BinaryOp.MUL,
        Literal(4.0),
        center,
    )

    laplacian = BinaryExpr(
        BinaryOp.SUB,
        sum_all,
        four_center,
    )

    op = Operator(
        statements=(
            Assign(
                target=FieldAccess(out, (0, 0)),
                value=laplacian,
            ),
        ),
    )

    assert len(op.statements) == 1
    assert op.statements[0].target.field == out
    assert op.statements[0].target.field == out
    assert isinstance(op.statements[0].value, BinaryExpr)

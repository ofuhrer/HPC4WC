from yasmin.core import Dimension, float64
from yasmin.frontend import Expr, Field, Scalar
from yasmin.ir import stencil


def test_scalar_builds_expression() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    alpha = Scalar("alpha", dtype=float64)

    expr = alpha * u[0]

    assert isinstance(expr, Expr)
    assert isinstance(expr._ir, stencil.BinaryExpr)
    assert isinstance(expr._ir.lhs, stencil.ScalarRef)
    assert expr._ir.lhs.scalar == alpha._core

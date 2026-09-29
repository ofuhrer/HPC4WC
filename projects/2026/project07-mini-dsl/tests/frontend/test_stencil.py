from yasmin.core import Dimension, float64
from yasmin.frontend import Field, Stencil
from yasmin.ir import stencil as ir


def test_stencil_wraps_expression() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=float64)

    laplacian = u[-1, 0] + u[1, 0] + u[0, -1] + u[0, 1] - 4 * u[0, 0]

    stencil = Stencil(laplacian)

    assert stencil.expr == laplacian
    assert isinstance(stencil.expr._ir, ir.BinaryExpr)

import pytest

from yasmin.core import Dimension, float64
from yasmin.frontend import Expr, Field
from yasmin.ir import stencil


def test_build_field_expression() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=float64)

    laplacian = u[-1, 0] + u[1, 0] + u[0, -1] + u[0, 1] - 4 * u[0, 0]

    assert isinstance(laplacian, Expr)
    assert isinstance(laplacian._ir, stencil.BinaryExpr)


def test_field_access_validate_dimensionality() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=float64)

    with pytest.raises(ValueError):
        _ = u[0]

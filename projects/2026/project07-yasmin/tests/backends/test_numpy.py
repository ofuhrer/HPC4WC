import numpy as np

from yasmin.backends.numpy import NumPyBackend
from yasmin.core import Dimension, Field, Scalar, float64
from yasmin.ir.stencil import (
    Assign,
    BinaryExpr,
    BinaryOp,
    FieldAccess,
    Literal,
    Operator,
    ScalarRef,
)


def test_five_point_laplacian() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", (x, y), dtype=float64)
    out = Field("out", (x, y), dtype=float64)

    center = FieldAccess(u, (0, 0))
    left = FieldAccess(u, (-1, 0))
    right = FieldAccess(u, (1, 0))
    up = FieldAccess(u, (0, 1))
    down = FieldAccess(u, (0, -1))

    sum_horizontal = BinaryExpr(BinaryOp.ADD, left, right)
    sum_vertical = BinaryExpr(BinaryOp.ADD, up, down)
    sum_all = BinaryExpr(BinaryOp.ADD, sum_horizontal, sum_vertical)

    four_center = BinaryExpr(BinaryOp.MUL, Literal(4.0), center)

    laplacian = BinaryExpr(BinaryOp.SUB, sum_all, four_center)

    operator = Operator(
        statements=(
            Assign(
                target=FieldAccess(out, (0, 0)),
                value=laplacian,
            ),
        ),
    )

    u_data = np.arange(64, dtype=np.float64).reshape((8, 8))
    out_data = np.zeros_like(u_data)

    backend = NumPyBackend()
    backend.execute(
        operator,
        fields={
            u: u_data,
            out: out_data,
        },
    )

    expected = np.zeros_like(u_data)
    expected[1:-1, 1:-1] = (
        u_data[:-2, 1:-1]
        + u_data[2:, 1:-1]
        + u_data[1:-1, :-2]
        + u_data[1:-1, 2:]
        - 4 * u_data[1:-1, 1:-1]
    )

    np.testing.assert_allclose(out_data, expected)


def test_scalar_parameter() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)
    alpha = Scalar("alpha", dtype=float64)

    expression = BinaryExpr(
        BinaryOp.MUL,
        ScalarRef(alpha),
        FieldAccess(u, (0,)),
    )

    operator = Operator(
        statements=(
            Assign(
                target=FieldAccess(out, (0,)),
                value=expression,
            ),
        ),
    )

    u_data = np.arange(8, dtype=np.float64)
    out_data = np.zeros_like(u_data)

    backend = NumPyBackend()

    backend.execute(
        operator,
        fields={
            u: u_data,
            out: out_data,
        },
        scalars={
            alpha: 2.5,
        },
    )

    expected = 2.5 * u_data

    np.testing.assert_allclose(out_data, expected)


def test_asymmetric_stencil() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=float64)
    out = Field("out", dims=(x, y), dtype=float64)

    expression = BinaryExpr(
        BinaryOp.ADD,
        BinaryExpr(
            BinaryOp.ADD,
            FieldAccess(u, (-2, 0)),
            FieldAccess(u, (1, 0)),
        ),
        FieldAccess(u, (0, 3)),
    )

    operator = Operator(
        statements=(
            Assign(
                target=FieldAccess(out, (0, 0)),
                value=expression,
            ),
        ),
    )

    u_data = np.arange(64, dtype=np.float64).reshape((8, 8))
    out_data = np.zeros_like(u_data)

    backend = NumPyBackend()

    backend.execute(
        operator,
        fields={
            u: u_data,
            out: out_data,
        },
    )

    expected = np.zeros_like(u_data)
    expected[2:-1, :-3] = u_data[:-3, :-3] + u_data[3:, :-3] + u_data[2:-1, 3:]

    np.testing.assert_allclose(out_data, expected)

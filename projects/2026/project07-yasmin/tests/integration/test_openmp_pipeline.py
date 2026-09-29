import numpy as np
import pytest

import yasmin as yasi
from yasmin.frontend.expr import SymbolicExpr


@pytest.mark.openmp
def test_frontend_operator_with_openmp() -> None:
    x = yasi.Dimension("x")
    y = yasi.Dimension("y")

    u = yasi.Field("u", dims=(x, y), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x, y), dtype=yasi.float64)
    alpha = yasi.Scalar("alpha", dtype=yasi.float64)

    laplacian = yasi.Stencil(
        alpha * (u[-1, 0] + u[1, 0] + u[0, -1] + u[0, 1] - 4 * u[0, 0])
    )
    operator = yasi.Operator(target=out[0, 0], value=laplacian)

    u_data = np.arange(64, dtype=np.float64).reshape((8, 8)) ** 2
    u_out = np.zeros_like(u_data)

    alpha_value = 1.5

    yasi.execute(
        operator,
        backend="openmp",
        fields={
            u: u_data,
            out: u_out,
        },
        scalars={
            alpha: alpha_value,
        },
    )

    expected = np.zeros_like(u_data)
    expected[1:-1, 1:-1] = alpha_value * (
        u_data[:-2, 1:-1]
        + u_data[2:, 1:-1]
        + u_data[1:-1, :-2]
        + u_data[1:-1, 2:]
        - 4 * u_data[1:-1, 1:-1]
    )

    np.testing.assert_allclose(u_out, expected)


@pytest.mark.openmp
def test_openmp_operator_with_sequential_diffusion_assignments() -> None:
    x = yasi.Dimension("x")
    y = yasi.Dimension("y")

    u = yasi.Field("u", dims=(x, y), dtype=yasi.float64)
    u_new = yasi.Field("u_new", dims=(x, y), dtype=yasi.float64)
    alpha = yasi.Scalar("alpha", dtype=yasi.float64)

    laplace_u = u[-1, 0] + u[1, 0] + u[0, -1] + u[0, 1] - 4.0 * u[0, 0]

    laplace_u_new = (
        u_new[-1, 0] + u_new[1, 0] + u_new[0, -1] + u_new[0, 1] - 4.0 * u_new[0, 0]
    )

    op = yasi.Operator(
        (
            u_new[0, 0],
            u[0, 0] + alpha * laplace_u,
        ),
        (
            u[0, 0],
            u_new[0, 0] + alpha * laplace_u_new,
        ),
    )

    u_data = np.zeros((8, 8), dtype=np.float64)
    u_data[3:5, 3:5] = 1.0
    u_initial = u_data.copy()

    u_new_data = np.zeros_like(u_data)
    alpha_value = 0.1

    yasi.execute(
        op,
        backend="openmp",
        fields={
            u: u_data,
            u_new: u_new_data,
        },
        scalars={
            alpha: alpha_value,
        },
    )

    expected_u_new = np.zeros_like(u_initial)
    expected_u_new[1:-1, 1:-1] = u_initial[1:-1, 1:-1] + alpha_value * (
        u_initial[:-2, 1:-1]
        + u_initial[2:, 1:-1]
        + u_initial[1:-1, :-2]
        + u_initial[1:-1, 2:]
        - 4.0 * u_initial[1:-1, 1:-1]
    )

    expected_u = np.zeros_like(u_initial)
    expected_u[1:-1, 1:-1] = expected_u_new[1:-1, 1:-1] + alpha_value * (
        expected_u_new[:-2, 1:-1]
        + expected_u_new[2:, 1:-1]
        + expected_u_new[1:-1, :-2]
        + expected_u_new[1:-1, 2:]
        - 4.0 * expected_u_new[1:-1, 1:-1]
    )

    np.testing.assert_allclose(u_new_data, expected_u_new)
    np.testing.assert_allclose(u_data, expected_u)


@pytest.mark.openmp
def test_decorated_double_buffer_diffusion_with_openmp() -> None:
    x = yasi.Dimension("x")
    y = yasi.Dimension("y")

    u = yasi.Field("u", dims=(x, y), dtype=yasi.float64)
    u_new = yasi.Field("u_new", dims=(x, y), dtype=yasi.float64)
    alpha = yasi.Scalar("alpha", dtype=yasi.float64)

    @yasi.stencil
    def laplace(f: yasi.Field) -> SymbolicExpr:
        return f[-1, 0] + f[1, 0] + f[0, -1] + f[0, 1] - 4.0 * f[0, 0]

    @yasi.operator
    def diffuse(u: yasi.Field, u_new: yasi.Field, alpha: yasi.Scalar) -> None:
        u_new[0, 0] = u[0, 0] + alpha * laplace(u)
        u[0, 0] = u_new[0, 0] + alpha * laplace(u_new)

    op = diffuse(u, u_new, alpha)

    u_data = np.zeros((8, 8), dtype=np.float64)
    u_data[3:5, 3:5] = 1.0

    u_new_data = np.zeros_like(u_data)
    alpha_value = 0.1

    u_initial = u_data.copy()

    yasi.execute(
        op,
        backend="openmp",
        fields={
            u: u_data,
            u_new: u_new_data,
        },
        scalars={
            alpha: alpha_value,
        },
    )

    expected_u_new = np.zeros_like(u_initial)
    expected_u_new[1:-1, 1:-1] = u_initial[1:-1, 1:-1] + alpha_value * (
        u_initial[:-2, 1:-1]
        + u_initial[2:, 1:-1]
        + u_initial[1:-1, :-2]
        + u_initial[1:-1, 2:]
        - 4.0 * u_initial[1:-1, 1:-1]
    )

    expected_u = u_initial.copy()
    expected_u[1:-1, 1:-1] = expected_u_new[1:-1, 1:-1] + alpha_value * (
        expected_u_new[:-2, 1:-1]
        + expected_u_new[2:, 1:-1]
        + expected_u_new[1:-1, :-2]
        + expected_u_new[1:-1, 2:]
        - 4.0 * expected_u_new[1:-1, 1:-1]
    )

    np.testing.assert_allclose(u_new_data, expected_u_new)
    np.testing.assert_allclose(u_data, expected_u)


@pytest.mark.openmp
def test_3d_laplacian_with_cpp() -> None:
    x, y, z = yasi.Dimension("x", "y", "z")

    u = yasi.Field("u", dims=(x, y, z), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x, y, z), dtype=yasi.float64)

    @yasi.stencil
    def laplace_3d(f: yasi.Field) -> SymbolicExpr:
        return (
            f[-1, 0, 0]
            + f[1, 0, 0]
            + f[0, -1, 0]
            + f[0, 1, 0]
            + f[0, 0, -1]
            + f[0, 0, 1]
            - 6.0 * f[0, 0, 0]
        )

    @yasi.operator
    def apply(out: yasi.Field, u: yasi.Field) -> None:
        out[0, 0, 0] = laplace_3d(u)

    op = apply(out, u)

    u_data = np.zeros((6, 6, 6), dtype=np.float64)
    u_data[2:4, 2:4, 2:4] = 1.0
    out_data = np.zeros_like(u_data)

    yasi.execute(
        op,
        backend="openmp",
        fields={
            u: u_data,
            out: out_data,
        },
    )

    expected = np.zeros_like(u_data)
    expected[1:-1, 1:-1, 1:-1] = (
        u_data[:-2, 1:-1, 1:-1]
        + u_data[2:, 1:-1, 1:-1]
        + u_data[1:-1, :-2, 1:-1]
        + u_data[1:-1, 2:, 1:-1]
        + u_data[1:-1, 1:-1, :-2]
        + u_data[1:-1, 1:-1, 2:]
        - 6.0 * u_data[1:-1, 1:-1, 1:-1]
    )

    np.testing.assert_allclose(out_data, expected)

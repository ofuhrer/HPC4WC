import yasmin as yasi
from yasmin.frontend.expr import SymbolicExpr


def test_stencil_decorator_returns_stencil() -> None:
    x = yasi.Dimension("x")
    y = yasi.Dimension("y")

    u = yasi.Field("u", dims=(x, y), dtype=yasi.float64)

    @yasi.stencil
    def laplace(f: yasi.Field) -> SymbolicExpr:
        return f[-1, 0] + f[1, 0] + f[0, -1] + f[0, 1] - 4.0 * f[0, 0]

    result = laplace(u)

    assert isinstance(result, yasi.Stencil)


def test_stencil_decorator_can_be_reused_for_multiple_fields() -> None:
    x = yasi.Dimension("x")

    u = yasi.Field("u", dims=(x,), dtype=yasi.float64)
    v = yasi.Field("v", dims=(x,), dtype=yasi.float64)

    @yasi.stencil
    def centered(f: yasi.Field) -> SymbolicExpr:
        return f[-1] + f[0] + f[1]

    stencil_u = centered(u)
    stencil_v = centered(v)

    assert isinstance(stencil_u, yasi.Stencil)
    assert isinstance(stencil_v, yasi.Stencil)
    assert stencil_u._as_ir() != stencil_v._as_ir()


def test_operator_decorator_captures_single_assignment() -> None:
    x = yasi.Dimension("x")

    u = yasi.Field("u", dims=(x,), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x,), dtype=yasi.float64)

    @yasi.operator
    def double(out: yasi.Field, u: yasi.Field) -> None:
        out[0] = 2.0 * u[0]

    result = double(out, u)
    operator_ir = result._as_ir()

    assert isinstance(result, yasi.Operator)
    assert len(operator_ir.statements) == 1
    assert operator_ir.statements[0].target.field == out._core


def test_operator_decorator_preserves_assignment_order() -> None:
    x = yasi.Dimension("x")

    u = yasi.Field("u", dims=(x,), dtype=yasi.float64)
    u_new = yasi.Field("u_new", dims=(x,), dtype=yasi.float64)

    @yasi.operator
    def step(u: yasi.Field, u_new: yasi.Field) -> None:
        u_new[0] = 2.0 * u[0]
        u[0] = 3.0 * u_new[0]

    result = step(u, u_new)
    operator_ir = result._as_ir()

    assert len(operator_ir.statements) == 2
    assert operator_ir.statements[0].target.field == u_new._core
    assert operator_ir.statements[1].target.field == u._core


def test_operator_decorator_accepts_literal_assignment() -> None:
    x = yasi.Dimension("x")

    u = yasi.Field("u", dims=(x,), dtype=yasi.float64)

    @yasi.operator
    def initialize(u: yasi.Field) -> None:
        u[0] = 1.0

    result = initialize(u)
    operator_ir = result._as_ir()

    assert len(operator_ir.statements) == 1

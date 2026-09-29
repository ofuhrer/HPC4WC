import numpy as np

from yasmin.backends.cpp import CppBackend
from yasmin.core import Dimension, Field, Scalar, float64, int32
from yasmin.ir import loop


def test_emit_one_dimensional_stencil() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)

    i = loop.Index("x")

    function = loop.Function(
        name="stencil_1d",
        fields=(out, u),
        scalars=(),
        body=(
            loop.For(
                index=i,
                lower=loop.Literal(1),
                upper=loop.BinaryExpr(
                    loop.BinaryOp.SUB,
                    loop.Extent(out, 0),
                    loop.Literal(1),
                ),
                body=(
                    loop.Store(
                        field=out,
                        indices=(i,),
                        value=loop.BinaryExpr(
                            loop.BinaryOp.ADD,
                            loop.Load(
                                u,
                                (
                                    loop.BinaryExpr(
                                        loop.BinaryOp.SUB,
                                        i,
                                        loop.Literal(1),
                                    ),
                                ),
                            ),
                            loop.Load(
                                u,
                                (
                                    loop.BinaryExpr(
                                        loop.BinaryOp.ADD,
                                        i,
                                        loop.Literal(1),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    source = CppBackend().source(function)

    assert "void stencil_1d" in source
    assert "for (int x = 1;" in source
    assert "out[x]" in source
    assert "u[(x - 1)]" in source
    assert "u[(x + 1)]" in source


def test_execute_compiled_cpp_stencil() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)

    i = loop.Index("x")

    function = loop.Function(
        name="stencil_1d",
        fields=(out, u),
        scalars=(),
        body=(
            loop.For(
                index=i,
                lower=loop.Literal(1),
                upper=loop.BinaryExpr(
                    loop.BinaryOp.SUB,
                    loop.Extent(out, 0),
                    loop.Literal(1),
                ),
                body=(
                    loop.Store(
                        field=out,
                        indices=(i,),
                        value=loop.BinaryExpr(
                            loop.BinaryOp.ADD,
                            loop.Load(
                                field=u,
                                indices=(
                                    loop.BinaryExpr(
                                        loop.BinaryOp.SUB,
                                        i,
                                        loop.Literal(1),
                                    ),
                                ),
                            ),
                            loop.Load(
                                field=u,
                                indices=(
                                    loop.BinaryExpr(
                                        loop.BinaryOp.ADD,
                                        i,
                                        loop.Literal(1),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    u_data = np.arange(16, dtype=np.float64)
    out_data = np.zeros_like(u_data)

    artifact = CppBackend().compile(function)

    artifact.function(
        fields={
            out: out_data,
            u: u_data,
        },
    )

    expected = np.zeros_like(u_data)
    expected[1:-1] = u_data[:-2] + u_data[2:]

    np.testing.assert_allclose(out_data, expected)


def test_cpp_emits_int32_types() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=int32)
    scale = Scalar("scale", dtype=int32)

    function = loop.Function(
        name="kernel",
        fields=(u,),
        scalars=(scale,),
        body=(),
    )

    source = CppBackend().source(function)

    assert "std::int32_t*" in source
    assert "std::int32_t" in source

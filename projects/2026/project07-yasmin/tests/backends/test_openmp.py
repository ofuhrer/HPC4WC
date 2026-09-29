import numpy as np
import pytest

from yasmin.backends.openmp import OpenMPBackend
from yasmin.compiler.config import OpenMPOptions
from yasmin.compiler.openmp import resolve_openmp_config
from yasmin.core import Dimension, Field, float64
from yasmin.ir import loop


def test_emit_openmp_parallel_loop() -> None:
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
                        value=loop.Load(
                            field=u,
                            indices=(i,),
                        ),
                    ),
                ),
            ),
        ),
    )

    config = resolve_openmp_config(
        function,
        options=OpenMPOptions(),
    )

    source = OpenMPBackend(
        config=config,
    ).source(function)

    assert "#pragma omp parallel for" in source
    assert "for (int x = 1;" in source


def test_parallelizes_only_outermost_loop() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=float64)
    out = Field("out", dims=(x, y), dtype=float64)

    i = loop.Index("x")
    j = loop.Index("y")

    inner = loop.For(
        index=j,
        lower=loop.Literal(1),
        upper=loop.BinaryExpr(
            loop.BinaryOp.SUB,
            loop.Extent(out, 1),
            loop.Literal(1),
        ),
        body=(
            loop.Store(
                field=out,
                indices=(i, j),
                value=loop.Load(
                    field=u,
                    indices=(i, j),
                ),
            ),
        ),
    )

    outer = loop.For(
        index=i,
        lower=loop.Literal(1),
        upper=loop.BinaryExpr(
            loop.BinaryOp.SUB,
            loop.Extent(out, 0),
            loop.Literal(1),
        ),
        body=(inner,),
    )

    function = loop.Function(
        name="copy_2d",
        fields=(out, u),
        scalars=(),
        body=(outer,),
    )

    config = resolve_openmp_config(
        function,
        options=OpenMPOptions(),
    )

    source = OpenMPBackend(
        config=config,
    ).source(function)

    assert source.count("#pragma omp parallel for") == 1
    assert "#pragma omp parallel for" in source
    assert "    for (int x = 1; x < (out_shape_0 - 1); ++x) {" in source
    assert "for (int y = 1; y < (out_shape_1 - 1); ++y) {" in source
    assert "out[(x * out_shape_1 + y)] = u[(x * u_shape_1 + y)];" in source


@pytest.mark.openmp
def test_execute_compiled_openmp_stencil() -> None:
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

    config = resolve_openmp_config(
        function,
        options=OpenMPOptions(),
    )

    artifact = OpenMPBackend(
        config=config,
    ).compile(function)

    artifact.function(
        fields={
            out: out_data,
            u: u_data,
        },
    )

    expected = np.zeros_like(u_data)
    expected[1:-1] = u_data[:-2] + u_data[2:]

    np.testing.assert_allclose(out_data, expected)

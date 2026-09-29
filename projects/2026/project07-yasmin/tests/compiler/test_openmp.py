import pytest

import yasmin.compiler.openmp as openmp_compiler
from yasmin.compiler.config import OpenMPOptions
from yasmin.compiler.openmp import resolve_openmp_config
from yasmin.core import Dimension, Field, float64
from yasmin.ir import loop


def _make_1d_function(
    *,
    field_name: str = "out",
) -> tuple[loop.Function, Field]:
    x = Dimension("x")
    out = Field(field_name, dims=(x,), dtype=float64)

    i = loop.Index("x")

    statement = loop.For(
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
                value=loop.Literal(0.0),
            ),
        ),
    )

    return (
        loop.Function(
            name="kernel",
            fields=(out,),
            scalars=(),
            body=(statement,),
        ),
        out,
    )


def _make_2d_function() -> tuple[loop.Function, Field]:
    x = Dimension("x")
    y = Dimension("y")

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
                value=loop.Literal(0.0),
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

    return (
        loop.Function(
            name="kernel",
            fields=(out,),
            scalars=(),
            body=(outer,),
        ),
        out,
    )


def test_resolve_generic_openmp_config() -> None:
    function, _ = _make_2d_function()

    config = resolve_openmp_config(
        function,
        options=OpenMPOptions(),
    )

    assert len(config.loop_configs) == 1

    loop_config = config.loop_configs[0]

    assert loop_config is not None
    assert loop_config.parallelize
    assert loop_config.collapse
    assert loop_config.num_threads is None
    assert loop_config.schedule == "static"
    assert loop_config.schedule_chunk is None


def test_resolve_adaptive_openmp_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openmp_compiler, "available_cores", lambda: 8)

    function, out = _make_2d_function()

    config = resolve_openmp_config(
        function,
        options=OpenMPOptions(),
        shapes={
            out: (4, 10_000),
        },
    )

    loop_config = config.loop_configs[0]

    assert loop_config is not None
    assert loop_config.parallelize
    assert loop_config.collapse
    assert loop_config.num_threads == 8


def test_resolve_small_loop_as_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openmp_compiler, "available_cores", lambda: 8)

    function, out = _make_1d_function()

    config = resolve_openmp_config(
        function,
        options=OpenMPOptions(),
        shapes={
            out: (512,),
        },
    )

    loop_config = config.loop_configs[0]

    assert loop_config is not None
    assert not loop_config.parallelize
    assert not loop_config.collapse
    assert loop_config.num_threads is None


def test_resolve_multiple_loop_configs_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openmp_compiler, "available_cores", lambda: 8)

    large_function, large = _make_1d_function(field_name="large")
    small_function, small = _make_1d_function(field_name="small")

    function = loop.Function(
        name="kernel",
        fields=(large, small),
        scalars=(),
        body=(
            large_function.body[0],
            small_function.body[0],
        ),
    )

    config = resolve_openmp_config(
        function,
        options=OpenMPOptions(),
        shapes={
            large: (10_000,),
            small: (512,),
        },
    )

    assert len(config.loop_configs) == 2

    large_config = config.loop_configs[0]
    small_config = config.loop_configs[1]

    assert large_config is not None
    assert small_config is not None

    assert large_config.parallelize
    assert large_config.num_threads == 8

    assert not small_config.parallelize
    assert small_config.num_threads is None

import numpy as np
import pytest

from yasmin.core import Dimension, Field, Scalar, float64
from yasmin.ir import loop
from yasmin.runtime.native import CompiledFunction


def _compiled_function() -> tuple[
    CompiledFunction,
    Field,
    Field,
    Scalar,
]:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)
    alpha = Scalar("alpha", dtype=float64)

    function = loop.Function(
        name="kernel",
        fields=(u, out),
        scalars=(alpha,),
        body=(),
    )

    compiled = object.__new__(CompiledFunction)
    compiled.function = function

    return compiled, u, out, alpha


def test_compiled_function_rejects_missing_field_binding() -> None:
    compiled, u, _, alpha = _compiled_function()

    with pytest.raises(
        ValueError,
        match="Missing field bindings: out",
    ):
        compiled._validate_bindings(
            fields={
                u: np.zeros(8, dtype=np.float64),
            },
            scalars={
                alpha: 0.1,
            },
        )


def test_compiled_function_rejects_unexpected_field_binding() -> None:
    compiled, u, out, alpha = _compiled_function()

    x = Dimension("x")
    extra = Field("extra", dims=(x,), dtype=float64)

    with pytest.raises(
        ValueError,
        match="Unexpected field bindings: extra",
    ):
        compiled._validate_bindings(
            fields={
                u: np.zeros(8, dtype=np.float64),
                out: np.zeros(8, dtype=np.float64),
                extra: np.zeros(8, dtype=np.float64),
            },
            scalars={
                alpha: 0.1,
            },
        )


def test_compiled_function_rejects_missing_scalar_binding() -> None:
    compiled, u, out, _ = _compiled_function()

    with pytest.raises(
        ValueError,
        match="Missing scalar bindings: alpha",
    ):
        compiled._validate_bindings(
            fields={
                u: np.zeros(8, dtype=np.float64),
                out: np.zeros(8, dtype=np.float64),
            },
            scalars={},
        )


def test_compiled_function_rejects_unexpected_scalar_binding() -> None:
    compiled, u, out, alpha = _compiled_function()

    beta = Scalar("beta", dtype=float64)

    with pytest.raises(
        ValueError,
        match="Unexpected scalar bindings: beta",
    ):
        compiled._validate_bindings(
            fields={
                u: np.zeros(8, dtype=np.float64),
                out: np.zeros(8, dtype=np.float64),
            },
            scalars={
                alpha: 0.1,
                beta: 0.2,
            },
        )

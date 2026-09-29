from collections.abc import Mapping
from typing import Any

import numpy.typing as npt

from yasmin.backends import NumPyBackend
from yasmin.compiler.compile import _compile_function
from yasmin.compiler.config import CompileConfig
from yasmin.frontend import Field, Operator, Scalar
from yasmin.lowering import lower

Array = npt.NDArray[Any]


def execute(
    operator: Operator,
    *,
    backend: str,
    fields: Mapping[Field, Array],
    scalars: Mapping[Scalar, int | float] | None = None,
) -> None:
    field_bindings = {field._core: value for field, value in fields.items()}
    scalar_bindings = {scalar._core: value for scalar, value in (scalars or {}).items()}

    operator_ir = operator._as_ir()

    if backend == "numpy":
        NumPyBackend().execute(
            operator=operator_ir,
            fields=field_bindings,
            scalars=scalar_bindings,
        )
        return

    if backend == "cpp":
        config = CompileConfig(
            backend="cpp",
        )
    elif backend == "openmp":
        config = CompileConfig(
            backend="openmp",
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    function = lower(
        operator=operator_ir,
        name="kernel",
    )

    shapes = {field: array.shape for field, array in field_bindings.items()}

    result = _compile_function(
        function,
        config=config,
        shapes=shapes,
    )

    result.artifact.function(
        fields=field_bindings,
        scalars=scalar_bindings,
    )

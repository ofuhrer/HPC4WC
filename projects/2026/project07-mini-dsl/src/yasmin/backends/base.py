from collections.abc import Mapping
from typing import Protocol, TypeVar

from yasmin.core import Field, Scalar
from yasmin.ir.stencil import Operator

FieldValue = TypeVar("FieldValue", contravariant=True)
ScalarValue = int | float


class ExecutionBackend(Protocol[FieldValue]):
    name: str

    def execute(
        self,
        operator: Operator,
        *,
        fields: Mapping[Field, FieldValue],
        scalars: Mapping[Scalar, ScalarValue] | None = None,
    ) -> None: ...

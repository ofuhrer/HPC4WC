from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy.typing as npt

from yasmin.compiler.config import CppOptions
from yasmin.compiler.openmp import OpenMPConfig
from yasmin.frontend import Field, Scalar
from yasmin.runtime.native import CompiledFunction

Array = npt.NDArray[Any]


@dataclass(frozen=True, slots=True)
class Kernel:
    _compiled: CompiledFunction
    source: str
    config: CppOptions | OpenMPConfig

    def __call__(
        self,
        *,
        fields: Mapping[Field, Array],
        scalars: Mapping[Scalar, int | float] | None = None,
    ) -> None:
        field_bindings = {field._core: array for field, array in fields.items()}
        scalar_bindings = {
            scalar._core: value for scalar, value in (scalars or {}).items()
        }

        self._compiled(
            fields=field_bindings,
            scalars=scalar_bindings,
        )

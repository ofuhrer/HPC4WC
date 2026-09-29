from dataclasses import dataclass

from yasmin.core import DType
from yasmin.core import Scalar as CoreScalar
from yasmin.frontend.expr import SymbolicExpr
from yasmin.ir import stencil


@dataclass(frozen=True, slots=True)
class Scalar(SymbolicExpr):
    _core: CoreScalar

    def __init__(
        self,
        name: str,
        *,
        dtype: DType,
    ) -> None:
        object.__setattr__(
            self,
            "_core",
            CoreScalar(name=name, dtype=dtype),
        )

    @property
    def name(self) -> str:
        return self._core.name

    @property
    def dtype(self) -> DType:
        return self._core.dtype

    def _as_ir(self) -> stencil.Expr:
        return stencil.ScalarRef(
            scalar=self._core,
        )

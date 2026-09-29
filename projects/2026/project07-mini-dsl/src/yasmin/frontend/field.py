from dataclasses import dataclass

from yasmin.core import Dimension, DType
from yasmin.core import Field as CoreField
from yasmin.frontend.expr import Expr, SymbolicExpr, _as_symbolic_expr
from yasmin.ir import stencil as stencil_ir


@dataclass(frozen=True, slots=True)
class Field:
    _core: CoreField

    def __init__(
        self,
        name: str,
        *,
        dims: tuple[Dimension, ...],
        dtype: DType,
    ) -> None:
        object.__setattr__(
            self,
            "_core",
            CoreField(
                name=name,
                dims=dims,
                dtype=dtype,
            ),
        )

    @property
    def name(self) -> str:
        return self._core.name

    @property
    def dims(self) -> tuple[Dimension, ...]:
        return self._core.dims

    @property
    def dtype(self) -> DType:
        return self._core.dtype

    def __getitem__(
        self,
        offsets: int | tuple[int, ...],
    ) -> Expr:
        normalized = (offsets,) if isinstance(offsets, int) else offsets

        if len(normalized) != len(self._core.dims):
            raise ValueError(
                f"Field {self._core.name!r} expects "
                f"{len(self._core.dims)} offsets, got {len(normalized)}"
            )

        return Expr(
            stencil_ir.FieldAccess(
                field=self._core,
                offsets=normalized,
            )
        )

    def __setitem__(
        self,
        offsets: int | tuple[int, ...],
        value: SymbolicExpr | int | float,
    ) -> None:
        from yasmin.frontend.capture import record_assignment

        target = self[offsets]

        record_assignment(
            target,
            _as_symbolic_expr(value),
        )

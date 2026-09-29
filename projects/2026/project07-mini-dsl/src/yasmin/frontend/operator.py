from dataclasses import dataclass
from typing import cast

from yasmin.frontend.expr import Expr, SymbolicExpr
from yasmin.ir import stencil as stencil_ir


@dataclass(frozen=True, slots=True)
class Assignment:
    target: Expr
    value: SymbolicExpr

    def __post_init__(self) -> None:
        target = self.target._as_ir()

        if not isinstance(target, stencil_ir.FieldAccess):
            raise TypeError(
                f"Assignment target must be a FieldAccess, got {type(target).__name__}"
            )

    def _as_ir(self) -> stencil_ir.Assign:
        target = cast(
            stencil_ir.FieldAccess,
            self.target._as_ir(),
        )

        return stencil_ir.Assign(
            target=target,
            value=self.value._as_ir(),
        )


AssignmentSpec = tuple[Expr, SymbolicExpr]


@dataclass(frozen=True, slots=True)
class Operator:
    _assignments: tuple[Assignment, ...]

    def __init__(
        self,
        target: Expr | tuple[Expr, SymbolicExpr],
        value: SymbolicExpr | tuple[Expr, SymbolicExpr] | None = None,
        *assignments: tuple[Expr, SymbolicExpr],
    ) -> None:
        normalized: tuple[Assignment, ...]

        # Operator(target=out[0], value=expr)
        # Operator(out[0], expr)
        if isinstance(target, Expr):
            if not isinstance(value, SymbolicExpr):
                raise TypeError("Operator value must be a symbolic expression")

            if assignments:
                raise TypeError("Cannot mix a single assignment with assignment tuples")

            normalized = (
                Assignment(
                    target=target,
                    value=value,
                ),
            )

        # Operator((out1[0], expr1), (out2[0], expr2), ...)
        else:
            specs: tuple[tuple[Expr, SymbolicExpr], ...]

            if value is None:
                specs = (target, *assignments)
            elif isinstance(value, tuple):
                specs = (target, value, *assignments)
            else:
                raise TypeError(
                    "Operator expects either "
                    "Operator(target, value) or "
                    "Operator((target, value), ...)"
                )

            normalized = tuple(
                Assignment(
                    target=assignment_target,
                    value=assignment_value,
                )
                for assignment_target, assignment_value in specs
            )

        object.__setattr__(
            self,
            "_assignments",
            normalized,
        )

    def _as_ir(self) -> stencil_ir.Operator:
        return stencil_ir.Operator(
            statements=tuple(assignment._as_ir() for assignment in self._assignments)
        )

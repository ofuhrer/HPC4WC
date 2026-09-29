from dataclasses import dataclass

from yasmin.ir.stencil.expr import Expr, FieldAccess


@dataclass(frozen=True, slots=True)
class Assign:
    target: FieldAccess
    value: Expr

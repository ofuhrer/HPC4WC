from dataclasses import dataclass

from yasmin.core import Field
from yasmin.ir.loop.expr import Expr, Index


class Stmt:
    pass


@dataclass(frozen=True, slots=True)
class Store(Stmt):
    field: Field
    indices: tuple[Expr, ...]
    value: Expr


@dataclass(frozen=True, slots=True)
class For(Stmt):
    index: Index
    lower: Expr
    upper: Expr
    body: tuple[Stmt, ...]

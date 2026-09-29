from dataclasses import dataclass
from enum import Enum

from yasmin.core import Field, Scalar


class Expr:
    pass


@dataclass(frozen=True, slots=True)
class Literal(Expr):
    value: int | float


@dataclass(frozen=True, slots=True)
class ScalarRef(Expr):
    scalar: Scalar


@dataclass(frozen=True, slots=True)
class Index(Expr):
    name: str


@dataclass(frozen=True, slots=True)
class Extent(Expr):
    field: Field
    dim: int


@dataclass(frozen=True, slots=True)
class Load(Expr):
    field: Field
    indices: tuple[Expr, ...]


class BinaryOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"


@dataclass(frozen=True, slots=True)
class BinaryExpr(Expr):
    op: BinaryOp
    lhs: Expr
    rhs: Expr

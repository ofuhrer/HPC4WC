from dataclasses import dataclass

from yasmin.core import Field, Scalar
from yasmin.ir.loop.stmt import Stmt


@dataclass(frozen=True, slots=True)
class Function:
    name: str
    fields: tuple[Field, ...]
    scalars: tuple[Scalar, ...]
    body: tuple[Stmt, ...]

from dataclasses import dataclass

from yasmin.ir.stencil.stmt import Assign


@dataclass(frozen=True, slots=True)
class Operator:
    statements: tuple[Assign, ...]

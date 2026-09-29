from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from yasmin.frontend.expr import Expr, SymbolicExpr


@dataclass
class AssignmentCapture:
    assignments: list[tuple[Expr, SymbolicExpr]] = field(default_factory=list)


_current_capture: ContextVar[AssignmentCapture | None] = ContextVar(
    "yasmin_assignment_capture",
    default=None,
)


def begin_capture() -> tuple[AssignmentCapture, Token[AssignmentCapture | None]]:
    capture = AssignmentCapture()
    token = _current_capture.set(capture)
    return capture, token


def end_capture(token: Token[AssignmentCapture | None]) -> None:
    _current_capture.reset(token)


def record_assignment(
    target: Expr,
    value: SymbolicExpr,
) -> None:
    capture = _current_capture.get()

    if capture is None:
        raise RuntimeError(
            "Field assignment is only allowed inside a @yasmin.operator function"
        )

    capture.assignments.append((target, value))

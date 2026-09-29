from collections.abc import Callable
from functools import wraps
from typing import ParamSpec

from yasmin.frontend.capture import begin_capture, end_capture
from yasmin.frontend.expr import SymbolicExpr
from yasmin.frontend.operator import Operator
from yasmin.frontend.stencil import Stencil

P = ParamSpec("P")


def stencil(
    func: Callable[P, SymbolicExpr],
) -> Callable[P, Stencil]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Stencil:
        return Stencil(func(*args, **kwargs))

    return wrapper


def operator(
    func: Callable[P, None],
) -> Callable[P, Operator]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Operator:
        capture, token = begin_capture()

        try:
            func(*args, **kwargs)
        finally:
            end_capture(token)

        if not capture.assignments:
            raise ValueError(
                f"Operator {func.__name__} must contain at least one assignment"
            )
        return Operator(*capture.assignments)

    return wrapper

from typing import overload

from yasmin.core import Dimension as CoreDimension


@overload
def Dimension(name: str, /) -> CoreDimension: ...


@overload
def Dimension(name: str, second: str, /, *rest: str) -> tuple[CoreDimension, ...]: ...


def Dimension(name: str, /, *names: str) -> CoreDimension | tuple[CoreDimension, ...]:
    if not names:
        return CoreDimension(name)

    return tuple(CoreDimension(name) for name in (name, *names))

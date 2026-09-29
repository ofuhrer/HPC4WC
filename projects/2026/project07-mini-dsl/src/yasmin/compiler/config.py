from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

from yasmin.core import Field

BackendName: TypeAlias = Literal["cpp", "openmp"]
FieldShapes: TypeAlias = Mapping[Field, tuple[int, ...]]


class BackendOptions:
    """Base type for backend-specific compilation options."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class CppOptions(BackendOptions):
    use_restrict: bool = True


@dataclass(frozen=True, slots=True)
class OpenMPOptions(CppOptions):
    use_collapse: bool = True
    schedule: str | None = "static"
    schedule_chunk: int | None = None
    num_threads: int | None = None
    adaptive: bool = True
    min_iters_per_thread: int = 1_000
    extra_compile_flags: tuple[str, ...] = (
        "-march=native",
        "-fno-math-errno",
    )

    @classmethod
    def baseline(cls) -> OpenMPOptions:
        return cls(
            use_collapse=False,
            schedule=None,
            num_threads=None,
            adaptive=False,
            extra_compile_flags=(),
        )


@dataclass(frozen=True, slots=True)
class CompileConfig:
    backend: BackendName
    options: BackendOptions | None = None

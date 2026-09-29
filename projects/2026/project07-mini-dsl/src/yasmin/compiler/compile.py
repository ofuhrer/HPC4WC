from dataclasses import dataclass
from typing import Literal

from yasmin.backends import CppBackend, OpenMPBackend
from yasmin.compiler.config import (
    BackendName,
    BackendOptions,
    CompileConfig,
    CppOptions,
    FieldShapes,
    OpenMPOptions,
)
from yasmin.compiler.kernel import Kernel
from yasmin.compiler.openmp import OpenMPConfig, resolve_openmp_config
from yasmin.frontend import Operator
from yasmin.ir import loop
from yasmin.lowering import lower
from yasmin.runtime.native import NativeArtifact

CppCacheKey = tuple[
    Literal["cpp"],
    loop.Function,
    CppOptions,
]

OpenMPCacheKey = tuple[
    Literal["openmp"],
    loop.Function,
    OpenMPConfig,
]

CompilationCacheKey = CppCacheKey | OpenMPCacheKey


@dataclass(frozen=True, slots=True)
class _CompilationResult:
    artifact: NativeArtifact
    config: CppOptions | OpenMPConfig


_compilation_cache: dict[
    CompilationCacheKey,
    _CompilationResult,
] = {}


def _cpp_options(options: BackendOptions | None) -> CppOptions:
    if options is None:
        return CppOptions()

    if isinstance(options, OpenMPOptions):
        raise TypeError("OpenMPOptions cannot be used with the C++ backend")

    if isinstance(options, CppOptions):
        return options

    raise TypeError(f"Unsupported C++ backend options: {type(options).__name__}")


def _openmp_options(options: BackendOptions | None) -> OpenMPOptions:
    if options is None:
        return OpenMPOptions()

    if isinstance(options, OpenMPOptions):
        return options

    raise TypeError(f"Unsupported OpenMP backend options: {type(options).__name__}")


def _compile_function(
    function: loop.Function,
    *,
    config: CompileConfig,
    shapes: FieldShapes | None = None,
) -> _CompilationResult:
    if config.backend == "cpp":
        options = _cpp_options(config.options)

        key: CompilationCacheKey = (
            "cpp",
            function,
            options,
        )

        cached = _compilation_cache.get(key)
        if cached is not None:
            return cached

        artifact = CppBackend(
            options=options,
        ).compile(function)

        result = _CompilationResult(
            artifact=artifact,
            config=options,
        )

    elif config.backend == "openmp":
        options = _openmp_options(config.options)

        openmp_config = resolve_openmp_config(
            function,
            options=options,
            shapes=shapes,
        )

        key = (
            "openmp",
            function,
            openmp_config,
        )

        cached = _compilation_cache.get(key)
        if cached is not None:
            return cached

        artifact = OpenMPBackend(
            config=openmp_config,
        ).compile(function)

        result = _CompilationResult(
            artifact=artifact,
            config=openmp_config,
        )

    else:
        raise ValueError(f"Unknown backend: {config.backend!r}")

    _compilation_cache[key] = result
    return result


def _clear_compilation_cache() -> None:
    _compilation_cache.clear()


def _compile_config(
    *,
    backend: BackendName | None,
    config: CompileConfig | None,
) -> CompileConfig:
    if config is not None:
        if backend is not None:
            raise ValueError("Specify either backend or config, not both")

        return config

    if backend is None:
        raise ValueError("Either backend or config must be specified")

    return CompileConfig(
        backend=backend,
    )


def compile(
    operator: Operator,
    *,
    backend: BackendName | None = None,
    config: CompileConfig | None = None,
) -> Kernel:
    compile_config = _compile_config(
        backend=backend,
        config=config,
    )

    function = lower(
        operator=operator._as_ir(),
        name="kernel",
    )

    result = _compile_function(
        function,
        config=compile_config,
    )

    return Kernel(
        _compiled=result.artifact.function,
        source=result.artifact.source,
        config=result.config,
    )

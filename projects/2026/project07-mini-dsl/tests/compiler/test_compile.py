from unittest.mock import MagicMock

import numpy as np
import pytest

import yasmin as yasi
from yasmin.backends.cpp import CppBackend
from yasmin.backends.openmp import OpenMPBackend
from yasmin.compiler.compile import _clear_compilation_cache
from yasmin.compiler.openmp import OpenMPConfig
from yasmin.runtime.native import NativeArtifact


def test_compile_cpp_returns_executable_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_compilation_cache()

    x = yasi.Dimension("x")

    u = yasi.Field("u", dims=(x,), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x,), dtype=yasi.float64)

    @yasi.operator
    def copy(u: yasi.Field, out: yasi.Field) -> None:
        out[0] = u[0]

    operator = copy(u, out)

    compiled = MagicMock()

    artifact = NativeArtifact(
        function=compiled,
        source="// generated source",
    )

    compile_mock = MagicMock(return_value=artifact)

    monkeypatch.setattr(
        CppBackend,
        "compile",
        compile_mock,
    )

    kernel = yasi.compile(
        operator,
        backend="cpp",
    )

    assert isinstance(kernel, yasi.Kernel)
    assert kernel.source == "// generated source"
    assert kernel.config == yasi.CppOptions()
    assert compile_mock.call_count == 1

    u_data = np.ones(16, dtype=np.float64)
    out_data = np.zeros_like(u_data)

    kernel(
        fields={
            u: u_data,
            out: out_data,
        },
    )

    assert compiled.call_count == 1

    call = compiled.call_args

    assert call.kwargs["fields"] == {
        u._core: u_data,
        out._core: out_data,
    }
    assert call.kwargs["scalars"] == {}


def test_compile_openmp_does_not_require_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_compilation_cache()

    x = yasi.Dimension("x")

    u = yasi.Field("u", dims=(x,), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x,), dtype=yasi.float64)

    @yasi.operator
    def copy(u: yasi.Field, out: yasi.Field) -> None:
        out[0] = u[0]

    operator = copy(u, out)

    compiled = MagicMock()

    artifact = NativeArtifact(
        function=compiled,
        source="// generated source",
    )

    compile_mock = MagicMock(return_value=artifact)

    monkeypatch.setattr(
        OpenMPBackend,
        "compile",
        compile_mock,
    )

    kernel = yasi.compile(
        operator,
        backend="openmp",
    )

    assert isinstance(kernel, yasi.Kernel)
    assert kernel.source == "// generated source"
    assert isinstance(kernel.config, OpenMPConfig)

    loop_config = kernel.config.loop_configs[0]

    assert loop_config is not None
    assert loop_config.parallelize
    assert loop_config.num_threads is None

    assert compile_mock.call_count == 1


def test_compile_rejects_backend_and_config() -> None:
    x = yasi.Dimension("x")
    u = yasi.Field("u", dims=(x,), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x,), dtype=yasi.float64)

    @yasi.operator
    def copy(u: yasi.Field, out: yasi.Field) -> None:
        out[0] = u[0]

    operator = copy(u, out)

    with pytest.raises(
        ValueError,
        match="Specify either backend or config, not both",
    ):
        yasi.compile(
            operator,
            backend="cpp",
            config=yasi.CompileConfig(backend="cpp"),
        )


def test_compile_requires_backend_or_config() -> None:
    x = yasi.Dimension("x")
    u = yasi.Field("u", dims=(x,), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x,), dtype=yasi.float64)

    @yasi.operator
    def copy(u: yasi.Field, out: yasi.Field) -> None:
        out[0] = u[0]

    operator = copy(u, out)

    with pytest.raises(
        ValueError,
        match="Either backend or config must be specified",
    ):
        yasi.compile(operator)


def test_compile_rejects_openmp_options_for_cpp() -> None:
    x = yasi.Dimension("x")
    u = yasi.Field("u", dims=(x,), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x,), dtype=yasi.float64)

    @yasi.operator
    def copy(u: yasi.Field, out: yasi.Field) -> None:
        out[0] = u[0]

    operator = copy(u, out)

    config = yasi.CompileConfig(
        backend="cpp",
        options=yasi.OpenMPOptions(),
    )

    with pytest.raises(
        TypeError,
        match="OpenMPOptions cannot be used with the C\\+\\+ backend",
    ):
        yasi.compile(
            operator,
            config=config,
        )


def test_compile_rejects_cpp_options_for_openmp() -> None:
    x = yasi.Dimension("x")
    u = yasi.Field("u", dims=(x,), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x,), dtype=yasi.float64)

    @yasi.operator
    def copy(u: yasi.Field, out: yasi.Field) -> None:
        out[0] = u[0]

    operator = copy(u, out)

    config = yasi.CompileConfig(
        backend="openmp",
        options=yasi.CppOptions(),
    )

    with pytest.raises(
        TypeError,
        match="Unsupported OpenMP backend options: CppOptions",
    ):
        yasi.compile(
            operator,
            config=config,
        )


def test_compile_and_execute_share_cpp_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_compilation_cache()

    x = yasi.Dimension("x")

    u = yasi.Field("u", dims=(x,), dtype=yasi.float64)
    out = yasi.Field("out", dims=(x,), dtype=yasi.float64)

    @yasi.operator
    def copy(u: yasi.Field, out: yasi.Field) -> None:
        out[0] = u[0]

    operator = copy(u, out)

    compiled = MagicMock()

    artifact = NativeArtifact(
        function=compiled,
        source="// generated source",
    )

    compile_mock = MagicMock(return_value=artifact)

    monkeypatch.setattr(
        CppBackend,
        "compile",
        compile_mock,
    )

    yasi.compile(
        operator,
        backend="cpp",
    )

    u_data = np.ones(32, dtype=np.float64)
    out_data = np.zeros_like(u_data)

    yasi.execute(
        operator,
        backend="cpp",
        fields={
            u: u_data,
            out: out_data,
        },
    )

    assert compile_mock.call_count == 1
    assert compiled.call_count == 1

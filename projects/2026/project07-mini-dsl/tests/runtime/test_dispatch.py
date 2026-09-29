from unittest.mock import MagicMock

import numpy as np
import pytest

import yasmin as yasi
import yasmin.compiler.openmp as openmp_compiler
from yasmin.backends.cpp import CppBackend
from yasmin.backends.openmp import OpenMPBackend
from yasmin.compiler.compile import _clear_compilation_cache
from yasmin.runtime.native import NativeArtifact


def test_cpp_compilation_cache_reuses_kernel_across_shapes(
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

    first_u = np.ones(16, dtype=np.float64)
    first_out = np.zeros_like(first_u)

    yasi.execute(
        operator,
        backend="cpp",
        fields={
            u: first_u,
            out: first_out,
        },
    )

    second_u = np.ones(64, dtype=np.float64)
    second_out = np.zeros_like(second_u)

    yasi.execute(
        operator,
        backend="cpp",
        fields={
            u: second_u,
            out: second_out,
        },
    )

    assert compile_mock.call_count == 1
    assert compiled.call_count == 2


def test_openmp_compilation_cache_reuses_equivalent_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_compilation_cache()

    monkeypatch.setattr(
        openmp_compiler,
        "available_cores",
        lambda: 4,
    )

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

    first_u = np.ones(8_000, dtype=np.float64)
    first_out = np.zeros_like(first_u)

    yasi.execute(
        operator,
        backend="openmp",
        fields={
            u: first_u,
            out: first_out,
        },
    )

    second_u = np.ones(10_000, dtype=np.float64)
    second_out = np.zeros_like(second_u)

    yasi.execute(
        operator,
        backend="openmp",
        fields={
            u: second_u,
            out: second_out,
        },
    )

    assert compile_mock.call_count == 1
    assert compiled.call_count == 2


def test_openmp_compilation_cache_separates_different_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_compilation_cache()

    monkeypatch.setattr(
        openmp_compiler,
        "available_cores",
        lambda: 4,
    )

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

    small_u = np.ones(512, dtype=np.float64)
    small_out = np.zeros_like(small_u)

    yasi.execute(
        operator,
        backend="openmp",
        fields={
            u: small_u,
            out: small_out,
        },
    )

    large_u = np.ones(8_000, dtype=np.float64)
    large_out = np.zeros_like(large_u)

    yasi.execute(
        operator,
        backend="openmp",
        fields={
            u: large_u,
            out: large_out,
        },
    )

    assert compile_mock.call_count == 2
    assert compiled.call_count == 2

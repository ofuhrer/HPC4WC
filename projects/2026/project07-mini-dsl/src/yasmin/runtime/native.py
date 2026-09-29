import ctypes
import os
import platform
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from yasmin.core import (
    DType,
    Field,
    Scalar,
    float32,
    float64,
    int32,
    int64,
)
from yasmin.ir import loop

Array = npt.NDArray[Any]
ScalarValue = int | float


def _ctypes_type(dtype: DType) -> type[Any]:
    if dtype == float32:
        return ctypes.c_float

    if dtype == float64:
        return ctypes.c_double

    if dtype == int32:
        return ctypes.c_int32

    if dtype == int64:
        return ctypes.c_int64

    raise TypeError(f"Unsupported native dtype: {dtype.name}")


def _numpy_dtype(dtype: DType) -> np.dtype[Any]:
    if dtype == float32:
        return np.dtype(np.float32)

    if dtype == float64:
        return np.dtype(np.float64)

    if dtype == int32:
        return np.dtype(np.int32)

    if dtype == int64:
        return np.dtype(np.int64)

    raise TypeError(f"Unsupported native dtype: {dtype.name}")


@dataclass(frozen=True, slots=True)
class SharedLibrary:
    library: ctypes.CDLL
    directory: tempfile.TemporaryDirectory[str]


def compile_cpp(
    source: str,
    *,
    compiler: str | None = None,
    extra_flags: tuple[str, ...] = (),
) -> SharedLibrary:
    directory = tempfile.TemporaryDirectory()
    root = Path(directory.name)

    source_path = root / "kernel.cpp"

    system = platform.system()

    shared_flags: tuple[str, ...]

    if system == "Darwin":
        library_path = root / "libkernel.dylib"
        shared_flags = ("-dynamiclib",)
    elif system == "Linux":
        library_path = root / "libkernel.so"
        shared_flags = ("-shared", "-fPIC")
    else:
        directory.cleanup()
        raise RuntimeError(f"Unsupported platform: {system}")

    source_path.write_text(source)

    compiler = compiler or os.environ.get("CXX") or "c++"

    command = [
        compiler,
        "-O3",
        "-std=c++17",
        *shared_flags,
        *extra_flags,
        str(source_path),
        "-o",
        str(library_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        directory.cleanup()
        raise RuntimeError(f"Compilation failed:\n{e.stderr}") from e
    except OSError as e:
        directory.cleanup()
        raise RuntimeError(f"Failed to run compiler: {e!r}") from e

    return SharedLibrary(
        library=ctypes.CDLL(str(library_path)),
        directory=directory,
    )


class CompiledFunction:
    def __init__(
        self,
        function: loop.Function,
        shared_library: SharedLibrary,
    ) -> None:
        self.function = function
        self.shared_library = shared_library

        native_function = getattr(
            shared_library.library,
            function.name,
        )

        native_function.restype = None

        argtypes: list[Any] = []

        for field in function.fields:
            ctype = _ctypes_type(field.dtype)
            argtypes.append(ctypes.POINTER(ctype))

        for field in function.fields:
            for _dim in field.dims:
                argtypes.append(ctypes.c_int)

        for scalar in function.scalars:
            argtypes.append(_ctypes_type(scalar.dtype))

        native_function.argtypes = argtypes

        self._function = native_function

    def __call__(
        self,
        *,
        fields: Mapping[Field, Array],
        scalars: Mapping[Scalar, ScalarValue] | None = None,
    ) -> None:
        scalar_bindings = scalars or {}

        self._validate_bindings(
            fields=fields,
            scalars=scalar_bindings,
        )

        args: list[Any] = []

        for field in self.function.fields:
            array = fields[field]
            expected_dtype = _numpy_dtype(field.dtype)

            if array.dtype != expected_dtype:
                raise TypeError(
                    f"Field {field.name!r} must have dtype "
                    f"{expected_dtype.name}, got {array.dtype}"
                )

            if not array.flags.c_contiguous:
                raise ValueError(f"Field {field.name!r} must be C-contiguous")

            if array.ndim != len(field.dims):
                raise ValueError(
                    f"Field {field.name!r} expects "
                    f"{len(field.dims)} dimensions, got {array.ndim}"
                )

            ctype = _ctypes_type(field.dtype)
            args.append(array.ctypes.data_as(ctypes.POINTER(ctype)))

        for field in self.function.fields:
            array = fields[field]

            for extent in array.shape:
                args.append(int(extent))

        for scalar in self.function.scalars:
            ctype = _ctypes_type(scalar.dtype)
            args.append(ctype(scalar_bindings[scalar]))

        self._function(*args)

    def _validate_bindings(
        self,
        *,
        fields: Mapping[Field, Array],
        scalars: Mapping[Scalar, ScalarValue],
    ) -> None:
        expected_fields = set(self.function.fields)
        provided_fields = set(fields)

        missing_fields = expected_fields - provided_fields
        if missing_fields:
            names = ", ".join(sorted(field.name for field in missing_fields))
            raise ValueError(f"Missing field bindings: {names}")

        unexpected_fields = provided_fields - expected_fields
        if unexpected_fields:
            names = ", ".join(sorted(field.name for field in unexpected_fields))
            raise ValueError(f"Unexpected field bindings: {names}")

        expected_scalars = set(self.function.scalars)
        provided_scalars = set(scalars)

        missing_scalars = expected_scalars - provided_scalars
        if missing_scalars:
            names = ", ".join(sorted(scalar.name for scalar in missing_scalars))
            raise ValueError(f"Missing scalar bindings: {names}")

        unexpected_scalars = provided_scalars - expected_scalars
        if unexpected_scalars:
            names = ", ".join(sorted(scalar.name for scalar in unexpected_scalars))
            raise ValueError(f"Unexpected scalar bindings: {names}")


@dataclass(frozen=True, slots=True)
class NativeArtifact:
    function: CompiledFunction
    source: str

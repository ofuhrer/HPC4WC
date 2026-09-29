# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import math
import types
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from icon4py.tools.py2fgen import _codegen, _definitions


try:
    import cupy as cp  # type: ignore[import-not-found]
except ImportError:
    cp = None

if TYPE_CHECKING:
    import cffi

C_STR_TYPE_TO_NP_DTYPE: Final[dict[str, np.dtype]] = {
    _codegen.BUILTIN_TO_CPP_TYPE[_definitions.FLOAT64]: np.dtype(np.float64),
    _codegen.BUILTIN_TO_CPP_TYPE[_definitions.FLOAT32]: np.dtype(np.float32),
    _codegen.BUILTIN_TO_CPP_TYPE[_definitions.BOOL]: np.dtype(np.bool_),
    _codegen.BUILTIN_TO_CPP_TYPE[_definitions.INT32]: np.dtype(np.int32),
    _codegen.BUILTIN_TO_CPP_TYPE[_definitions.INT64]: np.dtype(np.int64),
}


def _resolve_dtype(ffi: cffi.FFI, ptr: cffi.FFI.CData) -> np.dtype:
    c_type = ffi.getctype(ffi.typeof(ptr).item)
    dtype = C_STR_TYPE_TO_NP_DTYPE.get(c_type)
    if dtype is None:
        raise ValueError(f"Unsupported C data type: {c_type}")
    return dtype


def _unpack_numpy(ffi: cffi.FFI, ptr: cffi.FFI.CData, *sizes: int) -> np.typing.NDArray:
    """Return a NumPy view of a CFFI pointer, sharing memory with the Fortran allocation."""
    length = math.prod(sizes)
    dtype = _resolve_dtype(ffi, ptr)
    return np.frombuffer(ffi.buffer(ptr, length * dtype.itemsize), dtype=dtype).reshape(
        sizes, order="F"
    )


def _unpack_cupy(ffi: cffi.FFI, ptr: cffi.FFI.CData, *sizes: int) -> cp.ndarray:
    """Return a CuPy view of an OpenACC device pointer, sharing memory with the Fortran allocation."""
    assert cp is not None

    if not sizes:
        raise ValueError("Sizes must be provided to determine the array shape.")

    length = math.prod(sizes)
    dtype = _resolve_dtype(ffi, ptr)
    total_size = length * dtype.itemsize

    current_device = cp.cuda.Device()
    ptr_val = int(ffi.cast("uintptr_t", ptr))
    mem = cp.cuda.UnownedMemory(ptr_val, total_size, owner=ptr, device_id=current_device.id)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    return cp.ndarray(shape=sizes, dtype=dtype, memptr=memptr, order="F")


def unpack(
    xp: types.ModuleType, ffi: cffi.FFI, ptr: cffi.FFI.CData, *sizes: int
) -> np.typing.NDArray:
    if xp is np:
        return _unpack_numpy(ffi, ptr, *sizes)
    elif xp is cp:
        return _unpack_cupy(ffi, ptr, *sizes)
    else:
        raise ValueError(f"Unsupported array type: {xp}. Expected Numpy or CuPy.")


def as_array(ffi: cffi.FFI, array_info: _definitions.ArrayInfo) -> np.ndarray | None:  # or cupy
    """
    Utility function to convert an ArrayInfo to a NumPy or CuPy array.

    The returned array is a direct view of the Fortran-allocated memory; its
    element type is taken from the CFFI pointer.

    Args:
        ffi:        The CFFI FFI instance.
        array_info: The ArrayInfo object containing the pointer and shape information.
    """
    ptr, shape, on_gpu, is_optional = array_info
    xp = cp if on_gpu else np
    if ptr == ffi.NULL:
        if is_optional:
            return None
        else:
            raise RuntimeError("Parameter is not optional, but received 'NULL'.")
    return unpack(xp, ffi, ptr, *shape)


def default_mapping(
    _: Any, param_descriptor: _definitions.ParamDescriptor
) -> _definitions.MapperType | None:
    """
    Provide default mappings for raw Fortran data to Python data types.

    Array parameters are mapped from 'ArrayInfo's to NumPy/CuPy arrays.
    Scalar 'BOOL' is delivered by CFFI as a Python 'int' (the C side is
    `unsigned char`); convert it to 'bool' here.
    """
    if isinstance(param_descriptor, _definitions.ArrayParamDescriptor):
        # one mapper per parameter, maxsize=2 covers double-buffering identical calls
        @functools.lru_cache(maxsize=2)
        def array_mapper(
            array_info: _definitions.ArrayInfo, *, ffi: cffi.FFI
        ) -> _definitions.NDArray:
            return as_array(ffi, array_info)

        return array_mapper
    if (
        isinstance(param_descriptor, _definitions.ScalarParamDescriptor)
        and param_descriptor.dtype == _definitions.BOOL
    ):
        return lambda value, ffi: bool(value)
    return None

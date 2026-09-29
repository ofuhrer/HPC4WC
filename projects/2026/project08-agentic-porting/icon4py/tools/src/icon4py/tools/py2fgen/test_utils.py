# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import cffi
import numpy as np

from icon4py.tools.py2fgen import _codegen, _definitions


try:
    import cupy as cp  # type: ignore[import-not-found]

except ImportError:
    cp = None


"""
Utilities for testing a py2fgen code.

E.g. constructing 'array_infos' or translating 'array_infos' to arrays.
"""


def _from_np_dtype(dtype: np.dtype) -> _definitions.ScalarKind:
    dtype = np.dtype(dtype)
    if dtype == np.int32:
        return _definitions.ScalarKind.INT32
    elif dtype == np.int64:
        return _definitions.ScalarKind.INT64
    elif dtype == np.float32:
        return _definitions.ScalarKind.FLOAT32
    elif dtype == np.float64:
        return _definitions.ScalarKind.FLOAT64
    elif dtype == np.bool_:
        return _definitions.ScalarKind.BOOL
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def array_info(
    ptr: cffi.FFI.CData,
    shape: tuple[int, ...],
    on_gpu: bool,
    is_optional: bool,
) -> _definitions.ArrayInfo:
    return (ptr, shape, on_gpu, is_optional)


def array_to_array_info(
    arr: np.ndarray,  # or cp.ndarray
    *,
    ffi: cffi.FFI | None = None,
    keep_alive: bool = True,
    as_fortran_layout: bool = True,
) -> _definitions.ArrayInfo:
    """
    Utility for testing ArrayInfo.

    Takes a NumPy array and translates it to an 'ArrayInfo'.
    By default
    - the array is kept alive to avoid deallocation of the array before the pointer in 'ArrayInfo' is used;
    - the array is converted to Fortran layout.
    """
    on_gpu = not isinstance(arr, np.ndarray)
    xp = cp if on_gpu else np
    if ffi is None:
        ffi = cffi.FFI()
    if as_fortran_layout and not arr.flags["F_CONTIGUOUS"]:
        arr = xp.asfortranarray(arr)

    addr = arr.ctypes.data if not on_gpu else arr.data.ptr  # type: ignore[attr-defined] # we claim it's numpy while the 2nd case is for cupy
    strtype = _codegen.BUILTIN_TO_CPP_TYPE[_from_np_dtype(arr.dtype)]
    ptr = ffi.cast(f"{strtype}*", addr)

    if keep_alive:
        # bind the lifetime of the `arr` to `ptr`
        ptr = ffi.gc(ptr, lambda _=arr: None)  # type: ignore[misc] # cannot infer type of lambda

    return array_info(ptr, arr.shape, on_gpu, False)

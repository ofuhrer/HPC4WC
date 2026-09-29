# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from collections.abc import Callable
from typing import Any

import gt4py.next as gtx
import gt4py.next.custom_layout_allocators as gtx_allocators
import gt4py.next.typing as gtx_typing


try:
    import cupy as cp  # type: ignore[import-not-found]
except ImportError:
    cp = None


def is_cupy_device(allocator: gtx_typing.Allocator | None) -> bool:
    if allocator is None:
        return False

    if gtx.CUPY_DEVICE_TYPE is None:
        return False

    return gtx_allocators.is_field_allocation_tool_for(allocator, gtx.CUPY_DEVICE_TYPE)  # type: ignore [type-var] #gt4py-related typing


def sync(allocator: gtx_typing.Allocator | None = None) -> None:
    """
    Synchronize the device if appropriate for the given backend.

    Note: this is and ad-hoc interface, maybe the function should get the device to sync for.
    """
    # Type annotation already describes that only these types are allowed, but mypy coverage is not great.
    # The explicit assert avoids critical mistakes in using this function.
    assert allocator is None or gtx_allocators.is_field_allocation_tool(allocator)
    if allocator is not None and is_cupy_device(allocator):
        cp.cuda.runtime.deviceSynchronize()


def synchronized_function[**P, R](
    func: Callable[P, R], *, allocator: gtx_typing.Allocator | None
) -> Callable[P, R]:
    """
    Wraps a function and synchronizes after execution
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        result = func(*args, **kwargs)
        sync(allocator=allocator)
        return result

    return wrapper


def synchronized(
    allocator: gtx_typing.Allocator | None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that synchronizes the device after the function execution.
    """
    return functools.partial(synchronized_function, allocator=allocator)

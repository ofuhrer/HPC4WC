# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeAlias

import cffi
import numpy as np
from gt4py import eve
from gt4py.next.type_system import type_specifications as gtx_ts


# As long as we use gt4py.eve, we can also just use the gt4py.next types.
# Note that the user-facing type should be py2fgen.ScalarKind, so we can
# copy over the gt4py.next types if we want to.
ScalarKind: TypeAlias = gtx_ts.ScalarKind  # noqa: UP040

BOOL = gtx_ts.ScalarKind.BOOL
INT32 = gtx_ts.ScalarKind.INT32
INT64 = gtx_ts.ScalarKind.INT64
FLOAT32 = gtx_ts.ScalarKind.FLOAT32
FLOAT64 = gtx_ts.ScalarKind.FLOAT64


class MemorySpace(eve.StrEnum):
    """
    Host: The pointer is always a host pointer.
    MaybeDevice: If the Fortran code is compiled for OpenACC, the pointer will be a device pointer,
        otherwise it will be a host pointer.
    """

    HOST = "host"
    MAYBE_DEVICE = "maybe_device"


class ArrayParamDescriptor(eve.Node):
    """
    Describes an array parameter of a function.

    The information is used to generate the Fortran signature and semantics.
    Attributes:
        rank: The rank of the array.
        dtype: The data type of the array.
        memory_space: 'Host' or 'MaybeDevice', see ``MemorySpace``.
        is_optional: If True, the pointer can be NULL.
    """

    rank: int
    dtype: ScalarKind
    memory_space: MemorySpace
    is_optional: bool


class ScalarParamDescriptor(eve.Node):
    dtype: ScalarKind


ParamDescriptor: TypeAlias = ArrayParamDescriptor | ScalarParamDescriptor  # noqa: UP040
"""
Describes the parameter type of a function, which is used to generate the
Fortran signature and semantics.
"""

type ParamDescriptors = Mapping[str, ParamDescriptor]
"""
Mapping of parameter names to their descriptors.
"""


# cffi.FFI.CData is not available at runtime, therefore we provide a runtime
# alias with type `Any` (as the `TypeAlias` will be runtime evaluated)
if TYPE_CHECKING:
    type ArrayInfo = tuple[cffi.FFI.CData, tuple[int, ...], bool, bool]
    """
    ArrayInfo describes the runtime information of a buffer:

    Attributes:
        pointer: The CFFI pointer.
        shape: Shape of the buffer.
        on_gpu: If the ptr is for device memory (needs to be `False` if the ArrayParamDescriptor.memory_space is `Host`).
        is_optional: If True, the pointer can be NULL.

    Note: We use a plain tuple to minimize runtime overhead in the bindings.
    """
    # the docstring above is an unofficial pyright way of adding the docstring to TypeAlias,
    # however it doesn't work within TYPE_CHECKING, therefore tools will not pick it up
else:
    from typing import Any

    type ArrayInfo = tuple[Any, tuple[int, ...], bool, bool]

if TYPE_CHECKING:
    import cupy as cp  # type: ignore[import-not-found]

    type NDArray = cp.ndarray | np.ndarray
else:
    type NDArray = np.ndarray

type MapperType = (
    Callable[[ArrayInfo, cffi.FFI], Any]
    | Callable[[bool, cffi.FFI], Any]
    | Callable[[int, cffi.FFI], Any]
    | Callable[[float, cffi.FFI], Any]
)

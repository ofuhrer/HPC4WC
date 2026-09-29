# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging as log
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard, TypeVar

import array_api_compat
import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
import numpy.typing as npt

from icon4py.model.common import type_alias as ta
from icon4py.model.common.utils import device_utils


if TYPE_CHECKING:
    from icon4py.model.common.grid import base as grid_base
    from icon4py.model.common.states import utils as state_utils


try:
    import cupy as xp  # type: ignore[import-not-found]
except ImportError:
    import numpy as xp

ScalarT = TypeVar("ScalarT", bound=gtx_typing.Scalar)
NDArray: TypeAlias = (  # noqa: UP040
    np.ndarray[tuple[int, ...], np.dtype[ScalarT]] | xp.ndarray[tuple[int, ...], np.dtype[ScalarT]]
)
type NDArrayInterface = np.ndarray | xp.ndarray | gtx.Field

ScalarLikeArray: TypeAlias = (  # noqa: UP040
    np.ndarray[tuple[()], np.dtype[ScalarT]] | xp.ndarray[tuple[()], np.dtype[ScalarT]]
)


def is_ndarray(obj: Any) -> TypeGuard[NDArray]:
    return isinstance(obj, np.ndarray | xp.ndarray)


def is_rank0_ndarray(obj: Any) -> TypeGuard[ScalarLikeArray]:
    return is_ndarray(obj) and obj.shape == ()


def backend_name(backend: gtx_typing.Backend | None) -> str:
    return "embedded" if backend is None else backend.name


def as_numpy(array: NDArrayInterface) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, gtx.Field):
        return array.asnumpy()
    else:
        import cupy as cp  # noqa: PLC0415 [import-outside-top-level]

        return cp.asnumpy(array)


def array_ns(try_cupy: bool) -> ModuleType:
    if try_cupy:
        try:
            import cupy as cp  # noqa: PLC0415 [import-outside-top-level]

            return cp
        except ImportError:
            log.warning("No cupy installed, falling back to numpy for array_ns")
    import numpy as np  # noqa: PLC0415 [import-outside-top-level]

    return np


def import_array_ns(allocator: gtx_typing.Allocator | None) -> ModuleType:
    """Import cupy or numpy depending on a chosen GT4Py backend DevicType."""
    return array_ns(device_utils.is_cupy_device(allocator))


def scalar_like_array[ScalarT: gtx_typing.Scalar](
    value: ScalarT,
    allocator: ModuleType | gtx_typing.Allocator | None = None,
) -> ScalarLikeArray[ScalarT]:  # type: ignore[type-var] # ScalarT is a subtype of already specified other types
    """Create a 0-d array (scalar-like) with given value on specified array namespace or allocator."""
    array_ns = allocator if allocator in (np, xp) else import_array_ns(allocator)
    assert array_ns is not None and hasattr(array_ns, "asarray")
    return array_ns.asarray(value)


def as_field(
    field: gtx.Field,
    allocator: gtx_typing.Allocator | None = None,
    embedded_on_host: bool = False,
) -> gtx.Field:
    """Convenience function to transfer an existing Field to a given backend."""
    data = field.asnumpy() if embedded_on_host else field.ndarray
    return gtx.as_field(field.domain, data=data, allocator=allocator)  # type: ignore [arg-type] # type "ndarray[Any, Any] | NDArrayObject"; expected "NDArrayObject"


def random_field(
    grid: grid_base.Grid,
    *dims: gtx.Dimension,
    low: float = -1.0,
    high: float = 1.0,
    dtype: npt.DTypeLike | None = None,
    extend: dict[gtx.Dimension, int] | None = None,
    allocator: gtx_typing.Allocator | None = None,
) -> gtx.Field:
    arr = np.random.default_rng().uniform(
        low=low, high=high, size=_shape(grid, *dims, extend=extend)
    )
    if dtype:
        arr = arr.astype(dtype)
    return gtx.as_field(dims, arr, allocator=allocator)  # type: ignore [arg-type] # type "ndarray[Any, Any] | NDArrayObject"; expected "NDArrayObject"


def random_sign(
    grid: grid_base.Grid,
    *dims: gtx.Dimension,
    dtype: npt.DTypeLike | None = None,
    extend: dict[gtx.Dimension, int] | None = None,
    allocator: gtx_typing.Allocator | None = None,
) -> gtx.Field:
    """Generate a random field with values -1 or 1."""
    arr = np.random.default_rng().choice([-1, 1], size=_shape(grid, *dims, extend=extend))
    if dtype:
        arr = arr.astype(dtype)
    return gtx.as_field(dims, arr, allocator=allocator)  # type: ignore [arg-type] # type "ndarray[Any, Any] | NDArrayObject"; expected "NDArrayObject"


def random_mask(
    grid: grid_base.Grid,
    *dims: gtx.Dimension,
    dtype: npt.DTypeLike | None = None,
    extend: dict[gtx.Dimension, int] | None = None,
    allocator: gtx_typing.Allocator | None = None,
) -> gtx.Field:
    rng = np.random.default_rng()
    shape = _shape(grid, *dims, extend=extend)
    flat_arr = np.full(shape, False).flatten()
    num_true = int(flat_arr.size * 0.5)
    flat_arr[:num_true] = True
    rng.shuffle(flat_arr)
    arr = flat_arr.reshape(shape)
    if dtype:
        arr = arr.astype(dtype)
    return gtx.as_field(dims, arr, allocator=allocator)  # type: ignore [arg-type] # type "ndarray[Any, Any] | NDArrayObject"; expected "NDArrayObject"


def zero_field(
    grid: grid_base.Grid,
    *dims: gtx.Dimension,
    dtype: npt.DTypeLike = ta.wpfloat,
    extend: dict[gtx.Dimension, int] | None = None,
    allocator: gtx_typing.Allocator | None = None,
) -> gtx.Field:
    field_domain = {dim: (0, stop) for dim, stop in zip(dims, _shape(grid, *dims, extend=extend))}
    return gtx.constructors.zeros(field_domain, dtype=dtype, allocator=allocator)


def constant_field(
    grid: grid_base.Grid,
    value: float,
    *dims: gtx.Dimension,
    dtype: npt.DTypeLike = ta.wpfloat,
    allocator: gtx_typing.Allocator | None = None,
) -> gtx.Field:
    return gtx.as_field(
        dims,
        np.full(shape=tuple(map(lambda x: grid.size[x], dims)), fill_value=value, dtype=dtype),  # type: ignore [arg-type] # type "ndarray[Any, Any] | NDArrayObject"; expected "NDArrayObject"
        allocator=allocator,
    )


def _shape(
    grid: grid_base.Grid,
    *dims: gtx.Dimension,
    extend: dict[gtx.Dimension, int] | None = None,
) -> tuple[int, ...]:
    extend = extend or {}
    return tuple(grid.size[dim] + extend.get(dim, 0) for dim in dims)


def index_field(
    grid: grid_base.Grid,
    dim: gtx.Dimension,
    extend: dict[gtx.Dimension, int] | None = None,
    dtype: npt.DTypeLike = gtx.int32,
    allocator: gtx_typing.Allocator | None = None,
) -> gtx.Field:
    xp = import_array_ns(allocator)
    shapex = _shape(grid, dim, extend=extend)[0]
    return gtx.as_field((dim,), xp.arange(shapex, dtype=dtype), allocator=allocator)


def array_namespace(array: NDArray) -> ModuleType:
    """
    Returns the array namespace for a given array.
    """
    return array_api_compat.array_namespace(array)


def list2field(
    domain: gtx.Domain,
    values: NDArray,
    indices: tuple[NDArray, ...],
    default_value: state_utils.ScalarType,
    allocator: gtx_typing.Allocator,
) -> gtx.Field:
    if len(domain) != len(indices):
        raise RuntimeError("The number of indices must match the shape of the domain.")
    assert all(index.shape == indices[0].shape for index in indices if not isinstance(index, slice))
    xp = array_namespace(values)
    arr = xp.full(domain.shape, fill_value=default_value, dtype=values.dtype)
    arr[indices] = values
    return gtx.as_field(domain, arr, allocator=allocator)


def adjust_fortran_indices(inp: NDArray) -> NDArray:
    """For some Fortran arrays we need to subtract 1 to be compatible with Python indexing."""
    return inp - 1

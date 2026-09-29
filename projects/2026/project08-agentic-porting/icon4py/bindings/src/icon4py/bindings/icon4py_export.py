# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import types
import typing
from collections.abc import Callable, Sequence
from typing import Any, Union

import cffi
from gt4py import next as gtx
from gt4py.next import common as gtx_common
from gt4py.next.type_system import (
    type_specifications as ts,
    type_translation as gtx_type_translation,
)

from icon4py.tools import py2fgen


def _parse_type_spec(type_spec: ts.TypeSpec) -> tuple[list[gtx.Dimension], ts.ScalarKind]:
    if isinstance(type_spec, ts.ScalarType):
        return [], type_spec.kind
    elif isinstance(type_spec, ts.FieldType):
        assert isinstance(type_spec.dtype, ts.ScalarType)
        return type_spec.dims, type_spec.dtype.kind
    else:
        raise ValueError(f"Unsupported type specification: {type_spec}")


def _is_optional_type_hint(type_hint: Any) -> bool:
    return typing.get_origin(type_hint) in (
        types.UnionType,
        Union,
    ) and types.NoneType in typing.get_args(type_hint)


def _unpack_optional_type_hint(type_hint: Any) -> tuple[Any, bool]:
    if _is_optional_type_hint(type_hint):
        return next(arg for arg in typing.get_args(type_hint) if arg is not types.NoneType), True
    else:
        return type_hint, False


def _get_gt4py_type(type_hint: Any) -> tuple[ts.TypeSpec, bool] | None:
    non_optional_type, is_optional = _unpack_optional_type_hint(type_hint)
    try:
        return gtx_type_translation.from_type_hint(non_optional_type), is_optional
    except ValueError:
        return None


# TODO(egparedes): possibly use `TypeForm` for the annotation parameter,
# once https://peps.python.org/pep-0747/ is approved.
def field_annotation_descriptor_hook(annotation: Any) -> py2fgen.ParamDescriptor | None:
    """
    Translates GT4Py types to 'ParamDescriptor's.

    'gtx.Field' to 'ArrayParamDescriptor' and GT4Py scalars to 'ScalarParamDescriptor'.

    If types cannot be translated to GT4Py types, we delegate to the next mechanism (return 'None').
    """
    maybe_gt4py_type = _get_gt4py_type(annotation)
    if maybe_gt4py_type is None:
        return None

    gt4py_type, is_optional = maybe_gt4py_type
    dims, dtype = _parse_type_spec(gt4py_type)
    if dims:
        return py2fgen.ArrayParamDescriptor(
            rank=len(dims),
            dtype=dtype,
            memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
            is_optional=is_optional,
        )
    else:
        return py2fgen.ScalarParamDescriptor(dtype=dtype)


def _as_field(dims: Sequence[gtx.Dimension]) -> Callable:
    # in case the cache lookup is still performance relevant, we can replace it by a custom swap cache
    # (only for substitution mode where we know we have exactly 2 entries)
    # or by even marking fields as constant over the whole program run and immediately return on second call
    @functools.cache
    def impl(array_info: py2fgen.ArrayInfo, *, ffi: cffi.FFI) -> gtx.Field | None:
        arr = py2fgen.as_array(ffi, array_info)
        if arr is None:
            return None
        _, shape, _, _ = array_info
        domain = {d: s for d, s in zip(dims, shape, strict=True)}
        return gtx_common._field(arr, domain=gtx_common.domain(domain))

    return impl


def field_annotation_mapping_hook(
    annotation: Any, param_descriptor: py2fgen.ParamDescriptor
) -> Callable | None:
    """
    Translates 'ArrayInfo's to 'gtx.Field' if they are annotated with 'gtx.Field'.

    If the type is not a GT4Py type, we delegate to the default mappping (by returning 'None').
    """
    if not isinstance(param_descriptor, py2fgen.ArrayParamDescriptor):
        return None
    maybe_gt4py_type = _get_gt4py_type(annotation)
    if maybe_gt4py_type is None:
        return None
    gt4py_type, _ = maybe_gt4py_type
    dims, _ = _parse_type_spec(gt4py_type)
    return _as_field(dims)


export = py2fgen.export(
    annotation_descriptor_hook=field_annotation_descriptor_hook,
    annotation_mapping_hook=field_annotation_mapping_hook,
)

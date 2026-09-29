# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import typing
from typing import Optional

import cffi
import pytest
from gt4py import next as gtx
from gt4py.next.type_system import type_specifications as ts

from icon4py.bindings import icon4py_export
from icon4py.tools import py2fgen


export_with_mapping_hook = py2fgen.export(
    annotation_mapping_hook=icon4py_export.field_annotation_mapping_hook,
    param_descriptors={
        "a": py2fgen.ArrayParamDescriptor(
            rank=1,
            dtype=ts.ScalarKind.INT32,
            memory_space=py2fgen.MemorySpace.HOST,
            is_optional=False,
        ),
        "b": py2fgen.ScalarParamDescriptor(dtype=ts.ScalarKind.INT32),
    },
)

SomeDim = gtx.Dimension("SomeDim")


def make_array_info(
    ptr: cffi.FFI.CData,
    shape: tuple[int, ...],
    on_gpu: bool,
    is_optional: bool,
) -> py2fgen.ArrayInfo:
    return (ptr, shape, on_gpu, is_optional)


@export_with_mapping_hook
def foo(a: gtx.Field[gtx.Dims[SomeDim], gtx.int32], b: gtx.int32):
    return a, b


def test_mapping_hook():
    ffi = cffi.FFI()

    array_ptr = ffi.new("int[10]")

    result_a, result_b = foo(
        ffi=ffi,
        perf_counters={},
        a=make_array_info(shape=(10,), ptr=array_ptr, on_gpu=False, is_optional=False),
        b=5,
    )
    assert hasattr(result_a, "ndarray")
    assert result_b == 5


def fun_non_optional(_: int):
    pass


def fun_with_optional(_: int | None):
    pass


def fun_with_None(_: int | None):
    pass


def fun_with_None_first(_: None | int):
    pass


@pytest.mark.parametrize(
    "fun,is_optional",
    [
        (fun_non_optional, False),
        (fun_with_optional, True),
        (fun_with_None, True),
        (fun_with_None_first, True),
    ],
)
def test_is_optional_type_hint(fun, is_optional):
    testee = typing.get_type_hints(fun)["_"]

    result = icon4py_export._is_optional_type_hint(testee)

    assert result == is_optional


@pytest.mark.parametrize(
    "fun,expected_is_optional",
    [
        (fun_non_optional, False),
        (fun_with_optional, True),
        (fun_with_None, True),
        (fun_with_None_first, True),
    ],
)
def test_unpack_optional_type_hint(fun, expected_is_optional):
    testee = typing.get_type_hints(fun)["_"]

    result, is_optional = icon4py_export._unpack_optional_type_hint(testee)

    assert result is int
    assert is_optional == expected_is_optional

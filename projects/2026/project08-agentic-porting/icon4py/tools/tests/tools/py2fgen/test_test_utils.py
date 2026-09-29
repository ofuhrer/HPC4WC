# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gc

import cffi
import numpy as np
import pytest

from icon4py.tools import py2fgen
from icon4py.tools.py2fgen.test_utils import array_to_array_info


@pytest.fixture
def ffi():
    return cffi.FFI()


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.int64, np.float32, np.float64])
def test_array_as_array_info(ffi, dtype):
    testee = np.array([0, 1, 2, 3, 4], dtype=dtype)

    result = array_to_array_info(testee, ffi=ffi)

    assert isinstance(result, tuple)
    arr = py2fgen.as_array(ffi, result)
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, testee)


def test_array_as_array_info_lifetime(ffi):
    """
    Test that the passed array is kept alive by the 'array_to_array_info' function.
    """
    result = array_to_array_info(np.arange(100000, dtype=np.int32), ffi=ffi, keep_alive=True)

    gc.collect()  # Force garbage collection
    # If we are lucky the next allocation would give us the same memory in case it wasn't kept alive.
    # The reversed test was there and most of the time worked, but not reliably. So this test
    # is "most of the time" testing something, but sometimes not (but also will not fail).
    _dummy = np.arange(100000, dtype=np.int32) + 1

    arr = py2fgen.as_array(ffi, result)
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, np.arange(100000, dtype=np.int32))

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
from gt4py.next.embedded.nd_array_field import NdArrayField


try:
    import cupy as cp
except ImportError:
    cp = None


def compare_values_shallow(value1, value2, obj_name="value"):  # noqa: PLR0911, PLR0912
    # Handle comparison of NdArrayField objects
    if isinstance(value1, NdArrayField) and isinstance(value2, NdArrayField):
        try:
            np.testing.assert_equal(value1.asnumpy(), value2.asnumpy())
            return True, None
        except AssertionError:
            return False, f"Array mismatch for {obj_name}"

    # Handle comparison of dictionaries
    if isinstance(value1, dict) and isinstance(value2, dict):
        if value1.keys() != value2.keys():
            return False, f"Dict keys mismatch for {obj_name}: {value1.keys()} != {value2.keys()}"
        result, error_message = compare_objects(value1, value2, obj_name)
        if not result:
            return False, error_message

        return True, None

    # Handle comparison of tuples
    if isinstance(value1, tuple) and isinstance(value2, tuple):
        if len(value1) != len(value2):
            return False, f"Tuple length mismatch for {obj_name}: {len(value1)} != {len(value2)}"
        for index, (item1, item2) in enumerate(zip(value1, value2, strict=False)):
            result, error_message = compare_values_shallow(item1, item2, f"{obj_name}[{index}]")
            if not result:
                return False, error_message
        return True, None

    # Handle comparison of objects with attributes (__dict__)
    if hasattr(value1, "__dict__") and hasattr(value2, "__dict__"):
        result, error_message = compare_objects(value1, value2, obj_name)
        if not result:
            return False, error_message
        return True, None

    # Check if both values are instances of numpy scalar types
    if isinstance(value1, np.ScalarType) and isinstance(value2, np.ScalarType):
        if value1 != value2:
            return False, f"Value mismatch for {obj_name}: {value1} != {value2}"
        return True, None

    # Handle comparison of numpy array objects
    if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
        try:
            np.testing.assert_equal(value1, value2)  # Compare arrays for equality
            return True, None
        except AssertionError:
            return False, f"Array mismatch for {obj_name}"

    # Handle comparison of cupy array objects
    if cp is not None:
        if isinstance(value1, cp.ndarray) and isinstance(value2, cp.ndarray):
            try:
                cp.testing.assert_equal(value1, value2)  # Compare arrays for equality
                return True, None
            except AssertionError:
                return False, f"Array mismatch for {obj_name}"

    # Direct comparison for other types
    if value1 != value2:
        return False, f"Value mismatch for {obj_name}: {value1} != {value2}"
    else:
        return True, None


def compare_objects(obj1, obj2, obj_name="object"):  # noqa: PLR0911
    # Check if both objects are instances of numpy scalar types
    if isinstance(obj1, np.ScalarType) and isinstance(obj2, np.ScalarType):
        if obj1 != obj2:
            return False, f"Value mismatch for {obj_name}: {obj1} != {obj2}"
        return True, None

    # Check if both objects are lists
    if isinstance(obj1, list) and isinstance(obj2, list):
        # Check if lists have the same length
        if len(obj1) != len(obj2):
            return False, f"Length mismatch for {obj_name}: {len(obj1)} != {len(obj2)}"

        # Compare each element in the lists
        for index, (item1, item2) in enumerate(zip(obj1, obj2, strict=False)):
            result, error_message = compare_objects(item1, item2, f"{obj_name}[{index}]")
            if not result:
                return False, error_message
        return True, None

    # Check if both objects are instances of the same class
    if obj1.__class__ != obj2.__class__:
        return False, f"Class mismatch for {obj_name}: {obj1.__class__} != {obj2.__class__}"

    # Shallowly compare the attributes of both objects
    for attr, value in vars(obj1).items():
        other_value = getattr(obj2, attr, None)
        result, error_message = compare_values_shallow(value, other_value, f"{obj_name}.{attr}")
        if not result:
            return False, error_message

    return True, None

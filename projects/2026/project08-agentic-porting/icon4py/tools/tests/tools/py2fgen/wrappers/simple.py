# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""@py2fgen.export-decorated test functions."""

from __future__ import annotations

import numpy as np

from icon4py.tools import py2fgen


_FLOAT64_2D = py2fgen.ArrayParamDescriptor(
    rank=2,
    dtype=py2fgen.FLOAT64,
    memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
    is_optional=False,
)

_BOOL_1D = py2fgen.ArrayParamDescriptor(
    rank=1,
    dtype=py2fgen.BOOL,
    memory_space=py2fgen.MemorySpace.HOST,
    is_optional=False,
)

_BOOL_SCALAR = py2fgen.ScalarParamDescriptor(dtype=py2fgen.BOOL)


# A bare ``annotation_mapping_hook`` that always returns ``None`` — this falls
# back to ``_conversion.default_mapping``, which translates the raw
# ``ArrayInfo`` tuple py2fgen passes from CFFI into a NumPy view.
def _default_only(_: object, __: py2fgen.ParamDescriptor) -> None:
    return None


@py2fgen.export(
    param_descriptors={"inp": _FLOAT64_2D, "result": _FLOAT64_2D},
    annotation_mapping_hook=_default_only,
)
def square_from_function(inp: np.ndarray, result: np.ndarray) -> None:
    np.square(inp, out=result)


@py2fgen.export(
    param_descriptors={"inp": _FLOAT64_2D, "result": _FLOAT64_2D},
    annotation_mapping_hook=_default_only,
)
def square_error(inp: np.ndarray, result: np.ndarray) -> None:
    raise Exception("Exception foo occurred")


@py2fgen.export(
    param_descriptors={"flag": _BOOL_SCALAR, "mask": _BOOL_1D},
    annotation_mapping_hook=_default_only,
)
def fill_mask(flag: bool, mask: np.ndarray) -> None:
    # Writes through the NumPy view back into the Fortran array, exercising
    # both scalar-bool passing and bool-array writeback.
    mask[:] = flag

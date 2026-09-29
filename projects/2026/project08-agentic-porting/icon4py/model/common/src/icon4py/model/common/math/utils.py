# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
General-purpose mathematical utility functions.

Contains typed wrappers around standard math operations for use in factories
and validation, and general-purpose GT4Py field operators.
"""

import math

import numpy as np
from gt4py import next as gtx
from gt4py.next import where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


def compute_sqrt(
    input_val: np.float64,
) -> np.float64:
    """
    Compute the square root of input_val.
    math.sqrt is not sufficiently typed for the validation happening in the factories.
    """
    return np.float64(math.sqrt(input_val))


@gtx.field_operator
def invert_edge_field(f: fa.EdgeField[ta.wpfloat]) -> fa.EdgeField[ta.wpfloat]:
    """
    Invert values.
    Args:
        f: values

    Returns:
        1/f where f is not zero.
    """
    return where(f != 0.0, 1.0 / f, f)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_inverse_on_edges(
    f: fa.EdgeField[ta.wpfloat],
    f_inverse: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    invert_edge_field(f, out=f_inverse, domain={dims.EdgeDim: (horizontal_start, horizontal_end)})

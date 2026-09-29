# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Annotated, Any

import numpy as np
import pytest

from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import _export


def test_from_annotated():
    testee = Annotated[int, py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32)]

    result = _export._from_annotated(testee)

    assert isinstance(result, py2fgen.ScalarParamDescriptor)
    assert result.dtype == py2fgen.INT32


@pytest.mark.parametrize(
    "testee, expected",
    [
        (float, py2fgen.ScalarParamDescriptor(dtype=py2fgen.FLOAT32)),
        (np.float32, py2fgen.ScalarParamDescriptor(dtype=py2fgen.FLOAT32)),
        (int, ValueError),  # no descriptor deducible
        (
            Annotated[int, py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32)],
            py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32),
        ),
        (
            Annotated[int, py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32), str],
            py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32),
        ),
    ],
)
def test_get_param_descriptor_from_annotation(testee, expected):
    def float_param_descriptor_hook(annotation: Any):
        if annotation in (float, np.float32):
            return py2fgen.ScalarParamDescriptor(dtype=py2fgen.FLOAT32)
        return None

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _export.param_descriptor_from_annotation(
                testee, annotation_descriptor_hook=float_param_descriptor_hook
            )
    else:
        result = _export.param_descriptor_from_annotation(
            testee, annotation_descriptor_hook=float_param_descriptor_hook
        )
        assert result == expected

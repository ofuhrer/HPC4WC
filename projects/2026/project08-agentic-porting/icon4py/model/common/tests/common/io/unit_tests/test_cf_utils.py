# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common.io import cf_utils

from .. import utils as test_io_utils


@pytest.mark.parametrize("input_", test_io_utils.state_values())
def test_to_canonical_dim_order(input_):
    input_dims = input_.dims
    output = cf_utils.to_canonical_dim_order(input_)
    assert output.dims == (input_dims[1], input_dims[0])

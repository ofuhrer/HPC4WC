# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

import pytest


pytest.importorskip(
    "xarray",
    reason="Optional icon4py-common[io] dependencies are missing. Please install them before running tests.",
)

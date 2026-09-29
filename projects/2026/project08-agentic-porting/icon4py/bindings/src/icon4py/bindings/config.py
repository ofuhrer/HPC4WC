# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common.utils import env


WAIT_FOR_COMPILATION: bool = env.flag_to_bool("ICON4PY_WAIT_FOR_COMPILATION", False)
"""Wait in granule initialization until jit compilation is complete."""

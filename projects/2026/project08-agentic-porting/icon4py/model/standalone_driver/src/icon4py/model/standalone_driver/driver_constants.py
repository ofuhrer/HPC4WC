# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common import type_alias as ta


#: Factor multiplied to the user-defined CFL number to determine the whether to enter watchmode
CFL_ENTER_WATCHMODE_FACTOR = ta.wpfloat("0.81")

#: Threshold factor multiplied to the user-defined CFL number to maintain the number of substeps
CFL_THRESHOLD_FACTOR = ta.wpfloat("0.9")

#: Factor multiplied to the user-defined CFL number to determine the whether to leave watchmode
CFL_LEAVE_WATCHMODE_FACTOR = ta.wpfloat("0.76")

#: Adjustment factor for second order divergence damping
ADJUST_FACTOR_FOR_SECOND_ORDER_DIVDAMP = ta.wpfloat("0.8")

#: Initial period for second order divergence damping
INITIAL_PERIOD_FOR_SECOND_ORDER_DIVDAMP = ta.wpfloat("1800.0")

#: Transition end period for second order divergence damping
TRANSITION_END_PERIOD_FOR_SECOND_ORDER_DIVDAMP = ta.wpfloat("7200.0")

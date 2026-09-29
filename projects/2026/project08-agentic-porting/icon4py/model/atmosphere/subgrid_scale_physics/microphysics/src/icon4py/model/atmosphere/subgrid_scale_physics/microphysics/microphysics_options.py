# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import enum

import gt4py.next as gtx


class LiquidAutoConversionType(gtx.int32, enum.Enum):
    """
    Options for computing liquid auto conversion rate
    """

    #: Kessler (1969) liquid auto conversion mode
    KESSLER = 0
    #: Seifert & Beheng (2006) liquid auto conversion mode
    SEIFERT_BEHENG = 1


class SnowInterceptParameterization(gtx.int32, enum.Enum):
    """
    Options for deriving snow intercept parameter
    """

    #: Estimated intercept parameter for the snow size distribution from the best-fit line in figure 10(a) of Field et al. (2005)
    FIELD_BEST_FIT_ESTIMATION = 1
    #: Estimated intercept parameter for the snow size distribution from the general moment equation in table 2 of Field et al. (2005)
    FIELD_GENERAL_MOMENT_ESTIMATION = 2

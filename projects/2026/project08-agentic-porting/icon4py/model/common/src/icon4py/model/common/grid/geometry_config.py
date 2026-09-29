# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""Configuration options for the ICON grid geometry."""

import dataclasses
import typing

from icon4py.model.common.config import options as common_conf_opt


@dataclasses.dataclass(kw_only=True)
class GeometryConfig:
    """
    Contains parameters to configure the computation of grid geometry fields.
    """

    use_analytical_means: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description=(
                "If True, compute mean geometry values (mean cell area, mean dual area, "
                "mean edge length, mean dual edge length) analytically from the grid "
                "parameters instead of using global reductions."
            ),
        ),
    ] = True

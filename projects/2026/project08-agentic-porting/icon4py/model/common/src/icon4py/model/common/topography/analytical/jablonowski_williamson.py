# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, ClassVar

from icon4py.model.common import constants as phy_const, dimension as dims
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from icon4py.model.common.grid import grid_manager as gm


@dataclasses.dataclass
class JablonowskiWilliamsonConfig:
    u0: float = 35.0
    eta_0: float = 0.252
    # The default values are from mo_nh_jabw_exp.f90 and mo_nh_testcases_nml.f90

    fortran_name_map: ClassVar[dict[str, str]] = {"jw_u0": "u0"}


def jablonowski_williamson(
    *,
    config: JablonowskiWilliamsonConfig,
    grid_manager: gm.GridManager,
) -> data_alloc.NDArray:
    """Compute the JW surface geopotential height on cell centres.

    Implements the analytical mountain described in Jablonowski & Williamson
    (2006), eq. (9), adapted for the ICON grid.
    """
    u0 = config.u0
    eta_0 = config.eta_0
    cell_lat = grid_manager.coordinates[dims.CellDim]["lat"].ndarray

    array_ns = data_alloc.array_namespace(cell_lat)
    sin_lat = array_ns.sin(cell_lat)
    cos_lat = array_ns.cos(cell_lat)

    fac1 = u0 * array_ns.cos((1.0 - eta_0) * (math.pi / 2)) ** 1.5
    fac2 = (-2.0 * (sin_lat**6) * (cos_lat**2 + 1.0 / 3.0) + 1.0 / 6.3) * fac1
    fac3 = (
        (1.6 * (cos_lat**3) * (sin_lat**2 + 2.0 / 3.0) - 0.5 * (math.pi / 2))
        * phy_const.EARTH_RADIUS
        * phy_const.EARTH_ANGULAR_VELOCITY
    )
    return fac1 * (fac2 + fac3) / phy_const.GRAV

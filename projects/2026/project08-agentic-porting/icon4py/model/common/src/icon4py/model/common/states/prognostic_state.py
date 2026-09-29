# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.states.tracer_state import TracerConfig, TracerState
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid


@dataclasses.dataclass
class PrognosticState:
    """Class that contains the prognostic state.

    Corresponds to ICON t_nh_prog
    """

    rho: fa.CellKField[ta.wpfloat]  # density, rho(nproma, nlev, nblks_c) [kg/m^3]
    w: fa.CellKField[ta.wpfloat]  # vertical_wind field, w(nproma, nlevp1, nblks_c) [m/s]
    vn: fa.EdgeKField[
        ta.wpfloat
    ]  # horizontal wind normal to edges, vn(nproma, nlev, nblks_e)  [m/s]
    exner: fa.CellKField[ta.wpfloat]  # exner function, exner(nrpoma, nlev, nblks_c)
    theta_v: fa.CellKField[ta.wpfloat]  # virtual temperature, (nproma, nlev, nlbks_c) [K]
    tracer: TracerState = dataclasses.field(
        default_factory=TracerState
    )  # tracer concentration, one CellKField per active tracer [kg/kg]

    @property
    def w_1(self) -> fa.CellField[ta.wpfloat]:
        return self.w[dims.KDim(0)]


def initialize_prognostic_state(
    grid: icon_grid.IconGrid,
    allocator: gtx_typing.Allocator,
    tracer_config: TracerConfig | None = None,
) -> PrognosticState:
    """Initialize the prognostic state with zero fields."""
    if tracer_config is None:
        tracer_config = TracerConfig.none()
    rho = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    w = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    vn = data_alloc.zero_field(
        grid,
        dims.EdgeDim,
        dims.KDim,
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    exner = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    theta_v = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    tracer = TracerState(
        **{
            name: data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
            )
            for name in tracer_config.active_names
        }
    )
    return PrognosticState(rho=rho, w=w, vn=vn, exner=exner, theta_v=theta_v, tracer=tracer)

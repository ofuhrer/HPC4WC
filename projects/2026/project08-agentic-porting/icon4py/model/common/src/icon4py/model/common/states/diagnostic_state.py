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

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid


@dataclasses.dataclass
class DiagnosticState:
    """Class that contains the diagnostic state which is not used in dycore but may be used in physics.
    These variables are also stored for output purpose.

    Corresponds to ICON t_nh_diag
    """

    #: air pressure [Pa] at cell center and full levels, originally defined as pres in ICON
    pressure: fa.CellKField[ta.wpfloat]
    #: air pressure [Pa] at cell center and half levels, originally defined as pres_ifc and pres_sfc for surface pressure in ICON.
    pressure_ifc: fa.CellKField[ta.wpfloat]
    #: air temperature [K] at cell center, originally defined as temp in ICON
    temperature: fa.CellKField[ta.wpfloat]
    #: air virtual temperature [K] at cell center, originally defined as tempv in ICON
    virtual_temperature: fa.CellKField[ta.wpfloat]
    #: zonal wind speed [m/s] at cell center
    u: fa.CellKField[ta.wpfloat]
    #: meridional wind speed [m/s] at cell center
    v: fa.CellKField[ta.wpfloat]

    @property
    def surface_pressure(self) -> fa.CellField[ta.wpfloat]:
        return gtx.as_field((dims.CellDim,), self.pressure_ifc.ndarray[:, -1])


@dataclasses.dataclass
class DiagnosticMetricState:
    """Class that contains the diagnostic metric state for computing the diagnostic state."""

    ddqz_z_full: fa.CellKField[ta.wpfloat]
    rbf_vec_coeff_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat]
    rbf_vec_coeff_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat]


def initialize_diagnostic_state(
    grid: icon_grid.IconGrid,
    allocator: gtx_typing.Allocator,
) -> DiagnosticState:
    """Initialize the diagnostic state with zero fields."""
    pressure = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    pressure_ifc = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    temperature = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    virtual_temperature = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    u = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    v = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    return DiagnosticState(
        pressure=pressure,
        pressure_ifc=pressure_ifc,
        temperature=temperature,
        virtual_temperature=virtual_temperature,
        u=u,
        v=v,
    )

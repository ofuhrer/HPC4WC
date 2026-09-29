# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Host-side orchestration of the hydrostatic pressure diagnosis.

Wraps the ``diagnose_surface_pressure`` + ``diagnose_pressure`` stencils into the
sequence ICON performs in ``diagnose_pres_temp`` (mo_nh_diagnose_pres_temp.f90):
the surface pressure is extrapolated from the lowest three levels, then the
pressure is obtained by vertical integration of the virtual temperature.

This is the *hydrostatic* pressure, deliberately not the Exner-function shortcut
``P0REF * exner ** CPD_O_RD``. The moist initial conditions rely on the
hydrostatic value because ICON re-diagnoses pressure via ``diagnose_pres_temp``
when initializing the water vapour (``init_nh_inwp_tracers`` with
``l_rediag=.TRUE.``), so the converged/serialized state matches this integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.diagnostic_calculations.stencils import (
    diagnose_pressure as diagnose_pressure_stencil,
    diagnose_surface_pressure as diagnose_surface_pressure_stencil,
)
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common import field_type_aliases as fa
    from icon4py.model.common.grid import base as grid_base


def diagnose_pressure_surface_to_top(
    *,
    grid: grid_base.Grid,
    backend: gtx_typing.Backend | None,
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
    surface_pressure: fa.CellField[ta.wpfloat],
    pressure: fa.CellKField[ta.wpfloat],
    pressure_on_cells_half_levels: fa.CellKField[ta.wpfloat],
) -> None:
    """Diagnose the hydrostatic pressure into caller-provided buffers.

    Args:
        grid, backend: grid and gt4py backend.
        exner, virtual_temperature, ddqz_z_full: input cell-K fields.
        surface_pressure: cell field receiving the surface pressure.
        pressure: cell-K field receiving the full-level pressure.
        pressure_on_cells_half_levels: K-extended (``nlev+1``) output buffer for
            pressure on cell half-levels; also receives the diagnosed surface pressure.
    """
    num_levels = grid.num_levels
    cell_domain = h_grid.domain(dims.CellDim)
    horizontal_end = grid.end_index(cell_domain(h_grid.Zone.END))

    diagnose_surface_pressure_stencil.diagnose_surface_pressure.with_backend(backend)(
        exner=exner,
        virtual_temperature=virtual_temperature,
        ddqz_z_full=ddqz_z_full,
        surface_pressure=pressure_on_cells_half_levels,
        horizontal_start=0,
        horizontal_end=horizontal_end,
        vertical_start=num_levels,
        vertical_end=num_levels + 1,
        offset_provider={},
    )
    # surface pressure lives at the bottom interface; extract it as a cell field
    surface_pressure.ndarray[:] = pressure_on_cells_half_levels.ndarray[:, num_levels]
    pressure_on_cells_half_levels.ndarray[:, -1] = surface_pressure.ndarray

    diagnose_pressure_stencil.diagnose_pressure.with_backend(backend)(
        ddqz_z_full=ddqz_z_full,
        virtual_temperature=virtual_temperature,
        surface_pressure=surface_pressure,
        pressure=pressure,
        pressure_ifc=pressure_on_cells_half_levels,
        horizontal_start=0,
        horizontal_end=horizontal_end,
        vertical_start=0,
        vertical_end=num_levels,
        offset_provider={},
    )


def diagnose_pressure_surface_to_top_ndarray(
    *,
    grid: grid_base.Grid,
    backend: gtx_typing.Backend | None,
    allocator: gtx_typing.Allocator,
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
) -> data_alloc.NDArray:
    """Allocate work buffers, diagnose the hydrostatic pressure, return its ndarray.

    Convenience wrapper around :func:`diagnose_pressure_surface_to_top` for one-shot callers
    (e.g. initial-condition setup) that do not keep their own buffers.
    """
    surface_pressure = data_alloc.zero_field(
        grid, dims.CellDim, allocator=allocator, dtype=ta.wpfloat
    )
    pressure = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
    )
    pressure_on_cells_half_levels = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator, dtype=ta.wpfloat
    )
    diagnose_pressure_surface_to_top(
        grid=grid,
        backend=backend,
        exner=exner,
        virtual_temperature=virtual_temperature,
        ddqz_z_full=ddqz_z_full,
        surface_pressure=surface_pressure,
        pressure=pressure,
        pressure_on_cells_half_levels=pressure_on_cells_half_levels,
    )
    return pressure.ndarray

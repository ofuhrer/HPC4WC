# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.advection import (
    advection_horizontal,
    advection_states,
    advection_vertical,
)
from icon4py.model.atmosphere.advection.stencils.apply_density_increment import (
    apply_density_increment,
)
from icon4py.model.atmosphere.advection.stencils.apply_interpolated_tracer_time_tendency import (
    apply_interpolated_tracer_time_tendency,
)
from icon4py.model.atmosphere.advection.stencils.copy_cell_kdim_field import copy_cell_kdim_field
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    type_alias as ta,
)
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.model_options import setup_program
from icon4py.model.common.utils import data_allocation as data_alloc, fortran_config


"""
Advection module ported from ICON mo_advection_stepping.f90.
"""

log = logging.getLogger(__name__)


class HorizontalAdvectionType(Enum):
    """
    Horizontal operator scheme for advection (originally ihadv_tracer).
    """

    #: no horizontal advection
    NO_ADVECTION = 0
    #: 2nd order MIURA with linear reconstruction
    LINEAR_2ND_ORDER = 2


class HorizontalAdvectionLimiter(Enum):
    """
    Limiter for horizontal advection operator (originally itype_hlimit).
    """

    #: no horizontal limiter
    NO_LIMITER = 0
    #: positive definite horizontal limiter
    POSITIVE_DEFINITE = 4


class VerticalAdvectionType(Enum):
    """
    Vertical operator scheme for advection (originally ivadv_tracer).
    """

    #: no vertical advection
    NO_ADVECTION = 0
    #: 1st order upwind
    UPWIND_1ST_ORDER = 1
    #: 3rd order PPM
    PPM_3RD_ORDER = 3


class VerticalAdvectionLimiter(Enum):
    """
    Limiter for vertical advection operator (originally itype_vlimit).
    """

    #: no vertical limiter
    NO_LIMITER = 0
    #: semi-monotonic vertical limiter
    SEMI_MONOTONIC = 1


@dataclasses.dataclass(frozen=True)
class AdvectionConfig:
    """
    Contains necessary parameters to configure an advection run.
    """

    horizontal_advection_type: HorizontalAdvectionType
    horizontal_advection_limiter: HorizontalAdvectionLimiter
    vertical_advection_type: VerticalAdvectionType
    vertical_advection_limiter: VerticalAdvectionLimiter

    @classmethod
    def from_fortran_dict(cls, atmo_dict: dict[str, Any], **overrides: Any) -> AdvectionConfig:
        transport_nml = atmo_dict["transport_nml"]
        return cls(
            horizontal_advection_type=HorizontalAdvectionType(
                fortran_config.list_to_value(transport_nml["ihadv_tracer"])
            ),
            horizontal_advection_limiter=HorizontalAdvectionLimiter(
                fortran_config.list_to_value(transport_nml["itype_hlimit"])
            ),
            vertical_advection_type=VerticalAdvectionType(
                fortran_config.list_to_value(transport_nml["ivadv_tracer"])
            ),
            vertical_advection_limiter=VerticalAdvectionLimiter(
                fortran_config.list_to_value(transport_nml["itype_vlimit"])
            ),
            **overrides,
        )


class Advection(ABC):
    """
    Runs one three-dimensional advection step.

    Missing advection-specific features:
        -tracer loops: currently the `run` method only advects one type of tracer at once
        -optional tendency output: depending on the physics package, opt_ddt_tracer_adv might be needed
        -maximum advection height: tracer-specific control over which levels are used for advection
    """

    @abstractmethod
    def run(
        self,
        *,
        diagnostic_state: advection_states.AdvectionDiagnosticState,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        """
        Run an advection step.

        Args:
            diagnostic_state: output argument, data class that contains diagnostic variables
            prep_adv: input argument, data class that contains precalculated advection fields
            p_tracer_now: input argument, field that contains current tracer mass fraction
            p_tracer_new: output argument, field that contains new tracer mass fraction
            dtime: input argument, the time step

        """
        ...


class NoAdvection(Advection):
    """Class that implements disabled three-dimensional advection."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        backend: gtx_typing.Backend | None,
        exchange: decomposition.ExchangeRuntime,
    ):
        log.debug("advection class init - start")

        # input arguments
        self._grid = grid
        self._backend = backend
        self._exchange = exchange

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        # stencils
        self._copy_cell_kdim_field = setup_program(
            backend=self._backend,
            program=copy_cell_kdim_field,
            horizontal_sizes={
                "horizontal_start": self._start_cell_nudging,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": self._grid.num_levels,
            },
            offset_provider=self._grid.connectivities,
        )

    def run(
        self,
        *,
        diagnostic_state: advection_states.AdvectionDiagnosticState,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("advection run - start")
        log.debug("communication of prep_adv cell field: mass_flx_ic - start")
        self._exchange.exchange(
            dims.CellDim, prep_adv.mass_flx_ic, stream=decomposition.DEFAULT_STREAM
        )
        log.debug("communication of prep_adv cell field: mass_flx_ic - end")

        log.debug("running stencil copy_cell_kdim_field - start")
        self._copy_cell_kdim_field(
            field_in=p_tracer_now,
            field_out=p_tracer_new,
        )
        log.debug("running stencil copy_cell_kdim_field - end")

        log.debug("advection run - end")


class GodunovSplittingAdvection(Advection):
    """Class that implements three-dimensional advection based on Godunov splitting."""

    def __init__(
        self,
        *,
        horizontal_advection: advection_horizontal.HorizontalAdvection,
        vertical_advection: advection_vertical.VerticalAdvection,
        grid: icon_grid.IconGrid,
        metric_state: advection_states.AdvectionMetricState,
        backend: gtx_typing.Backend | None,
        exchange: decomposition.ExchangeRuntime,
        even_timestep: bool = False,
    ):
        log.debug("advection class init - start")

        # input arguments
        self._horizontal_advection = horizontal_advection
        self._vertical_advection = vertical_advection
        self._grid = grid
        self._metric_state = metric_state
        self._backend = backend
        self._exchange = exchange
        self._even_timestep = even_timestep  # originally jstep_adv(:)%marchuk_order = 1

        # density fields
        #: intermediate density times cell thickness, includes either the horizontal or vertical advective density increment [kg/m^2]
        self._rhodz_ast2 = data_alloc.zero_field(
            self._grid,
            dims.CellDim,
            dims.KDim,
            allocator=model_backends.get_allocator(self._backend),
        )
        self._determine_local_domains()
        # stencils
        self._apply_density_increment = setup_program(
            backend=self._backend,
            program=apply_density_increment,
            constant_args={
                "deepatmo_divzl": self._metric_state.deepatmo_divzl,
                "deepatmo_divzu": self._metric_state.deepatmo_divzu,
            },
            horizontal_sizes={
                "horizontal_end": self._end_cell_end,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self._apply_interpolated_tracer_time_tendency = setup_program(
            backend=self._backend,
            program=apply_interpolated_tracer_time_tendency,
            horizontal_sizes={
                "horizontal_start": self._start_cell_lateral_boundary,
                "horizontal_end": self._end_cell_lateral_boundary_level_4,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
        )

        log.debug("advection class init - end")

    def _determine_local_domains(self) -> None:
        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY)
        )
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._start_cell_lateral_boundary_level_3 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        self._end_cell_lateral_boundary_level_4 = self._grid.end_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self._end_cell_end = self._grid.end_index(cell_domain(h_grid.Zone.END))

    def run(
        self,
        *,
        diagnostic_state: advection_states.AdvectionDiagnosticState,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("advection run - start")

        log.debug("communication of prep_adv cell field: mass_flx_ic - start")
        self._exchange.exchange(
            dims.CellDim,
            prep_adv.mass_flx_ic,
            stream=decomposition.DEFAULT_STREAM,
        )
        log.debug("communication of prep_adv cell field: mass_flx_ic - end")

        # reintegrate density for conservation of mass
        rhodz_in, horizontal_start = (
            (diagnostic_state.airmass_now, self._start_cell_lateral_boundary_level_2)
            if self._even_timestep
            else (diagnostic_state.airmass_new, self._start_cell_lateral_boundary_level_3)
        )

        log.debug("running stencil apply_density_increment - start")
        self._apply_density_increment(
            rhodz_in=rhodz_in,
            p_mflx_contra_v=prep_adv.mass_flx_ic,
            rhodz_out=self._rhodz_ast2,
            p_dtime=dtime,
            even_timestep=self._even_timestep,
            horizontal_start=horizontal_start,
        )
        log.debug("running stencil apply_density_increment - end")

        # Godunov splitting
        if self._even_timestep:
            # vertical transport
            self._vertical_advection.run(
                prep_adv=prep_adv,
                p_tracer_now=p_tracer_now,
                p_tracer_new=p_tracer_new,
                rhodz_now=diagnostic_state.airmass_now,
                rhodz_new=self._rhodz_ast2,
                p_mflx_tracer_v=diagnostic_state.vfl_tracer,
                dtime=dtime,
                even_timestep=self._even_timestep,
            )

            # horizontal transport
            self._horizontal_advection.run(
                prep_adv=prep_adv,
                p_tracer_now=p_tracer_new,
                p_tracer_new=p_tracer_new,
                rhodz_now=self._rhodz_ast2,
                rhodz_new=diagnostic_state.airmass_new,
                p_mflx_tracer_h=diagnostic_state.hfl_tracer,
                dtime=dtime,
            )

        else:
            # horizontal transport
            self._horizontal_advection.run(
                prep_adv=prep_adv,
                p_tracer_now=p_tracer_now,
                p_tracer_new=p_tracer_new,
                rhodz_now=diagnostic_state.airmass_now,
                rhodz_new=self._rhodz_ast2,
                p_mflx_tracer_h=diagnostic_state.hfl_tracer,
                dtime=dtime,
            )

            # vertical transport
            self._vertical_advection.run(
                prep_adv=prep_adv,
                p_tracer_now=p_tracer_new,
                p_tracer_new=p_tracer_new,
                rhodz_now=self._rhodz_ast2,
                rhodz_new=diagnostic_state.airmass_new,
                p_mflx_tracer_v=diagnostic_state.vfl_tracer,
                dtime=dtime,
                even_timestep=self._even_timestep,
            )

        # update lateral boundaries with interpolated time tendencies
        if self._grid.limited_area:
            log.debug("running stencil apply_interpolated_tracer_time_tendency - start")
            self._apply_interpolated_tracer_time_tendency(
                p_tracer_now=p_tracer_now,
                p_grf_tend_tracer=diagnostic_state.grf_tend_tracer,
                p_tracer_new=p_tracer_new,
                p_dtime=dtime,
            )
            log.debug("running stencil apply_interpolated_tracer_time_tendency - end")

        # exchange updated tracer values, originally happens only if iforcing /= inwp
        log.debug("communication of tracer advection field: p_tracer_new - start")
        self._exchange.exchange(
            dims.CellDim,
            p_tracer_new,
            stream=decomposition.DEFAULT_STREAM,
        )
        log.debug("communication of tracer advection field: p_tracer_new - end")

        # finalize step
        self._even_timestep = not self._even_timestep

        log.debug("advection run - end")


def convert_config_to_horizontal_vertical_advection(  # noqa: PLR0912 [too-many-branches]
    *,
    config: AdvectionConfig,
    grid: icon_grid.IconGrid,
    interpolation_state: advection_states.AdvectionInterpolationState,
    least_squares_state: advection_states.AdvectionLeastSquaresState,
    metric_state: advection_states.AdvectionMetricState,
    edge_params: grid_states.EdgeParams,
    cell_params: grid_states.CellParams,
    backend: gtx_typing.Backend | None,
    exchange: decomposition.ExchangeRuntime,
) -> tuple[advection_horizontal.HorizontalAdvection, advection_vertical.VerticalAdvection]:
    assert exchange is not None, "Exchange runtime must not be None."
    horizontal_limiter: advection_horizontal.HorizontalFluxLimiter | None
    match config.horizontal_advection_limiter:
        case HorizontalAdvectionLimiter.NO_LIMITER:
            horizontal_limiter = advection_horizontal.NoLimiter()
        case HorizontalAdvectionLimiter.POSITIVE_DEFINITE:
            horizontal_limiter = advection_horizontal.PositiveDefinite(
                grid=grid,
                interpolation_state=interpolation_state,
                backend=backend,
                exchange=exchange,
            )
        case _:
            raise NotImplementedError("Unknown horizontal advection limiter.")

    horizontal_advection: advection_horizontal.HorizontalAdvection
    match config.horizontal_advection_type:
        case HorizontalAdvectionType.NO_ADVECTION:
            horizontal_advection = advection_horizontal.NoAdvection(grid=grid, backend=backend)
        case HorizontalAdvectionType.LINEAR_2ND_ORDER:
            tracer_flux = advection_horizontal.SecondOrderMiura(
                grid=grid,
                least_squares_state=least_squares_state,
                horizontal_limiter=horizontal_limiter,
                backend=backend,
            )
            horizontal_advection = advection_horizontal.SemiLagrangian(
                tracer_flux=tracer_flux,
                grid=grid,
                interpolation_state=interpolation_state,
                metric_state=metric_state,
                edge_params=edge_params,
                cell_params=cell_params,
                backend=backend,
            )
        case _:
            raise NotImplementedError("Unknown horizontal advection type.")

    vertical_limiter: advection_vertical.VerticalLimiter
    match config.vertical_advection_limiter:
        case VerticalAdvectionLimiter.NO_LIMITER:
            vertical_limiter = advection_vertical.NoLimiter(grid=grid, backend=backend)
        case VerticalAdvectionLimiter.SEMI_MONOTONIC:
            vertical_limiter = advection_vertical.SemiMonotonicLimiter(grid=grid, backend=backend)
        case _:
            raise NotImplementedError("Unknown vertical advection limiter.")

    vertical_advection: advection_vertical.VerticalAdvection
    match config.vertical_advection_type:
        case VerticalAdvectionType.NO_ADVECTION:
            vertical_advection = advection_vertical.NoAdvection(grid=grid, backend=backend)
        case VerticalAdvectionType.UPWIND_1ST_ORDER:
            boundary_conditions = advection_vertical.NoFluxCondition(grid=grid, backend=backend)
            vertical_advection = advection_vertical.FirstOrderUpwind(
                boundary_conditions=boundary_conditions,
                grid=grid,
                metric_state=metric_state,
                backend=backend,
            )
        case VerticalAdvectionType.PPM_3RD_ORDER:
            boundary_conditions = advection_vertical.NoFluxCondition(grid=grid, backend=backend)
            vertical_advection = advection_vertical.PiecewiseParabolicMethod(
                boundary_conditions=boundary_conditions,
                vertical_limiter=vertical_limiter,
                grid=grid,
                metric_state=metric_state,
                backend=backend,
            )
        case _:
            raise NotImplementedError("Unknown vertical advection type.")

    return horizontal_advection, vertical_advection


def convert_config_to_advection(
    *,
    config: AdvectionConfig,
    grid: icon_grid.IconGrid,
    interpolation_state: advection_states.AdvectionInterpolationState,
    least_squares_state: advection_states.AdvectionLeastSquaresState,
    metric_state: advection_states.AdvectionMetricState,
    edge_params: grid_states.EdgeParams,
    cell_params: grid_states.CellParams,
    backend: gtx_typing.Backend | None,
    exchange: decomposition.ExchangeRuntime,
    even_timestep: bool = False,
) -> Advection:
    if (
        config.horizontal_advection_type == HorizontalAdvectionType.NO_ADVECTION
        and config.vertical_advection_type == VerticalAdvectionType.NO_ADVECTION
    ):
        # advection is disabled for all tracers
        return NoAdvection(grid=grid, backend=backend, exchange=exchange)

    horizontal_advection, vertical_advection = convert_config_to_horizontal_vertical_advection(
        config=config,
        grid=grid,
        interpolation_state=interpolation_state,
        least_squares_state=least_squares_state,
        metric_state=metric_state,
        edge_params=edge_params,
        cell_params=cell_params,
        backend=backend,
        exchange=exchange,
    )

    advection = GodunovSplittingAdvection(
        horizontal_advection=horizontal_advection,
        vertical_advection=vertical_advection,
        grid=grid,
        metric_state=metric_state,
        backend=backend,
        exchange=exchange,
        even_timestep=even_timestep,
    )

    return advection

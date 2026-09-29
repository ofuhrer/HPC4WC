# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import enum
import functools
import logging
import statistics
from typing import TYPE_CHECKING, NamedTuple

import devtools

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.advection import advection_states
from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common import dimension as dims, time, type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.interpolation.stencils import edge_2_cell_vector_rbf_interpolation
from icon4py.model.common.math.stencils import generic_math_operations as gt4py_math_op
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    static_fields,
)
from icon4py.model.common.states.tracer_state import TracerState
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import config as driver_config


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


log = logging.getLogger(__name__)


class DriverStates(NamedTuple):
    """
    Initialized states for the driver run.

    Attributes:
        prep_advection_prognostic: Fields collecting data for advection during the solve nonhydro timestep.
        solve_nonhydro_diagnostic: Initial state for solve_nonhydro diagnostic variables.
        diffusion_diagnostic: Initial state for diffusion diagnostic variables.
        tracer_advection_diagnostic: Initial state for tracer advection diagnostic variables.
        prep_tracer_advection_prognostic: Precalculated fields for tracer advection.
        prognostics: Initial state for prognostic variables (double buffered).
        diagnostic: Initial state for global diagnostic variables.
    """

    prep_advection_prognostic: dycore_states.PrepAdvection | None
    solve_nonhydro_diagnostic: dycore_states.DiagnosticStateNonHydro | None
    diffusion_diagnostic: diffusion_states.DiffusionDiagnosticState | None
    tracer_advection_diagnostic: advection_states.AdvectionDiagnosticState | None
    prep_tracer_advection_prognostic: advection_states.AdvectionPrepAdvState | None
    prognostics: common_utils.TimeStepPair[prognostics.PrognosticState]
    diagnostic: diagnostics.DiagnosticState


class ModelTimeVariables:
    """
    Runtime time/date variables derived from config at initialisation.
    """

    simulation_current_datetime: time.AbsoluteTime
    simulation_start_datetime: time.AbsoluteTime
    simulation_end_datetime: time.AbsoluteTime
    n_time_steps: time.NumTimeSteps
    dtime: time.RelativeTime
    ndyn_substeps_var: int
    max_ndyn_substeps: int
    elapsed_time_in_seconds: ta.wpfloat
    is_first_step_in_simulation: bool
    cfl_watch_mode: bool

    def __init__(self, config: driver_config.DriverConfig) -> None:
        self._init_from_config(config)

    def _init_from_config(self, config: driver_config.DriverConfig) -> None:
        self.simulation_start_datetime = config.start_of_simulation
        match config.end_of_simulation:
            case time.NumTimeSteps() as n:
                self.n_time_steps = n
                self.simulation_current_datetime = config.start_of_simulation
                self.simulation_end_datetime = config.start_of_simulation + n * config.dtime
            case time.RelativeTime() as relative:
                self.n_time_steps = int(relative / config.dtime)
                self.simulation_current_datetime = config.start_of_simulation
                self.simulation_end_datetime = config.start_of_simulation + relative
            case time.AbsoluteTime() as absolute:
                self.n_time_steps = int((absolute - config.start_of_simulation) / config.dtime)
                self.simulation_current_datetime = config.start_of_simulation
                self.simulation_end_datetime = absolute
        self.dtime = config.dtime
        self.elapsed_time_in_seconds = ta.wpfloat("0.0")
        self.ndyn_substeps_var = config.ndyn_substeps
        self.max_ndyn_substeps = config.ndyn_substeps + 7
        self.is_first_step_in_simulation = True
        self.cfl_watch_mode = False

        if self.n_time_steps <= 0:
            raise ValueError("n_time_steps must be positive.")

    @functools.cached_property
    def dtime_in_seconds(self) -> ta.wpfloat:
        return ta.wpfloat(self.dtime.total_seconds())

    @property
    def substep_timestep(self) -> ta.wpfloat:
        return ta.wpfloat(self.dtime_in_seconds / self.ndyn_substeps_var)

    def advance_simulation_datetime(self) -> None:
        self.simulation_current_datetime += self.dtime
        self.elapsed_time_in_seconds += self.dtime_in_seconds

    def update_ndyn_substeps(self, new_ndyn_substeps: int) -> None:
        self.ndyn_substeps_var = new_ndyn_substeps

    def update_cfl_watch_mode(self, mode: bool) -> None:
        self.cfl_watch_mode = mode

    def reset(self, config: driver_config.DriverConfig) -> None:
        """
        Re-initialize all time-integration-related runtime values from the given config.
        """
        self._init_from_config(config)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"simulation_current_datetime={self.simulation_current_datetime!r}, "
            f"n_time_steps={self.n_time_steps}, "
            f"dtime={self.dtime}, "
            f"is_first_step={self.is_first_step_in_simulation})"
        )


class DriverTimers(enum.Enum):
    SOLVE_NH_FIRST_STEP = "solve_nh_first_step"
    SOLVE_NH = "solve_nh"
    DIFFUSION_FIRST_STEP = "diffusion_first_step"
    DIFFUSION = "diffusion"


@dataclasses.dataclass
class TimerCollection:
    timer_names: dataclasses.InitVar[list[str]]
    timers: dict[str, devtools.Timer] = dataclasses.field(init=False)

    def __post_init__(self, timer_names: str | list[str]) -> None:
        self.timers = {}
        self.add_timers(timer_names)

    def add_timers(self, timer_names: str | list[str]) -> None:
        for timer in timer_names:
            assert timer not in self.timers, f"Timer '{timer}' is already defined."
            self.timers[timer] = devtools.Timer(timer, dp=6, verbose=False)

    def show_timer_report(
        self,
    ) -> None:
        log.info("===== ICON4Py timer report =====")
        table_titles = (
            f"|{'timer name':^30}|"
            f"{'no. of times called':^23}|"
            f"{'mean time (s)':^23}|"
            f"{'std. deviation (s)':^23}|"
            f"{'min time (s)':^23}|"
            f"{'max time (s)':^23}|"
        )
        log.info(table_titles)
        log.info("-" * len(table_titles))
        for timer_name, timer in self.timers.items():
            times = []
            for r in timer.results:
                if not r.finish:
                    r.capture()
                times.append(r.elapsed())
            if len(times) > 0:
                log.info(
                    f"|{timer_name:^30}|"
                    f"{len(times):^23}|"
                    f"{statistics.mean(times):^23.8f}|"
                    f"{statistics.stdev(times) if len(times) > 1 else 0:^23.8f}|"
                    f"{min(times):^23.8f}|"
                    f"{max(times):^23.8f}|"
                )
            else:
                log.info(
                    f"|{timer_name:^30}|{'not started':^23}|{'':^23}|{'':^23}|{'':^23}|{'':^23}|"
                )


def assemble_driver_states(
    *,
    grid: icon_grid.IconGrid,
    allocator: gtx_typing.Allocator,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
    static_fields: static_fields.StaticFieldFactories,
    prognostic_state_now: prognostics.PrognosticState,
    diagnostic_state: diagnostics.DiagnosticState,
    experiment_config: driver_config.ExperimentConfig,
) -> DriverStates:
    prognostic_state_next = prognostics.PrognosticState(
        vn=data_alloc.as_field(prognostic_state_now.vn, allocator=allocator),
        w=data_alloc.as_field(prognostic_state_now.w, allocator=allocator),
        exner=data_alloc.as_field(prognostic_state_now.exner, allocator=allocator),
        rho=data_alloc.as_field(prognostic_state_now.rho, allocator=allocator),
        theta_v=data_alloc.as_field(prognostic_state_now.theta_v, allocator=allocator),
        tracer=TracerState(
            **{
                tracer.name: data_alloc.as_field(tracer.field, allocator=allocator)
                for tracer in prognostic_state_now.tracer.active_fields()
            }
        ),
    )
    prognostic_states = common_utils.TimeStepPair(prognostic_state_now, prognostic_state_next)

    cell_domain = h_grid.domain(dims.CellDim)
    end_cell_lateral_boundary_level_2 = grid.end_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    end_cell_end = grid.end_index(cell_domain(h_grid.Zone.END))

    rbf_vec_coeff_c1 = static_fields.interpolation.get(interpolation_attributes.RBF_VEC_COEFF_C1)
    rbf_vec_coeff_c2 = static_fields.interpolation.get(interpolation_attributes.RBF_VEC_COEFF_C2)

    edge_2_cell_vector_rbf_interpolation.edge_2_cell_vector_rbf_interpolation.with_backend(backend)(
        p_e_in=prognostic_states.current.vn,
        ptr_coeff_1=rbf_vec_coeff_c1,
        ptr_coeff_2=rbf_vec_coeff_c2,
        p_u_out=diagnostic_state.u,
        p_v_out=diagnostic_state.v,
        horizontal_start=end_cell_lateral_boundary_level_2,
        horizontal_end=end_cell_end,
        vertical_start=0,
        vertical_end=grid.num_levels,
        offset_provider=grid.connectivities,
    )
    exchange.exchange(dims.CellDim, diagnostic_state.u, diagnostic_state.v)

    perturbed_exner = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator)
    gt4py_math_op.compute_difference_on_cell_k.with_backend(backend)(
        field_a=prognostic_states.current.exner,
        field_b=static_fields.metrics.get(metrics_attributes.EXNER_REF_MC),
        output_field=perturbed_exner,
        horizontal_start=0,
        horizontal_end=grid.num_cells,
        vertical_start=0,
        vertical_end=grid.num_levels,
        offset_provider={},
    )

    diffusion_enabled = experiment_config.diffusion is not None
    solve_nonhydro_enabled = experiment_config.nonhydrostatic is not None
    tracer_advection_enabled = experiment_config.tracer_advection is not None

    diffusion_diagnostic_state = (
        diffusion_states.initialize_diffusion_diagnostic_state(grid=grid, allocator=allocator)
        if diffusion_enabled
        else None
    )
    solve_nonhydro_diagnostic_state = (
        dycore_states.initialize_solve_nonhydro_diagnostic_state(
            perturbed_exner_at_cells_on_model_levels=perturbed_exner,
            grid=grid,
            allocator=allocator,
        )
        if solve_nonhydro_enabled
        else None
    )
    prep_adv = (
        dycore_states.initialize_prep_advection(grid=grid, allocator=allocator)
        if solve_nonhydro_enabled
        else None
    )
    tracer_advection_diagnostic_state = (
        advection_states.initialize_advection_diagnostic_state(grid=grid, allocator=allocator)
        if tracer_advection_enabled
        else None
    )
    prep_tracer_adv = (
        advection_states.AdvectionPrepAdvState(
            vn_traj=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator),
            mass_flx_me=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator),
            mass_flx_ic=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator),
        )
        if tracer_advection_enabled
        else None
    )

    return DriverStates(
        prep_advection_prognostic=prep_adv,
        solve_nonhydro_diagnostic=solve_nonhydro_diagnostic_state,
        prep_tracer_advection_prognostic=prep_tracer_adv,
        tracer_advection_diagnostic=tracer_advection_diagnostic_state,
        diffusion_diagnostic=diffusion_diagnostic_state,
        prognostics=prognostic_states,
        diagnostic=diagnostic_state,
    )

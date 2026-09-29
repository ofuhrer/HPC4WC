# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import datetime
import functools
import logging
import pathlib
import types
from collections.abc import Callable

import gt4py.next as gtx
from gt4py.next import config as gtx_config
from gt4py.next.instrumentation import metrics as gtx_metrics

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.advection import advection_states
from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common import (
    dimension as dims,
    initial_condition,
    model_backends,
    time,
    topography,
    type_alias as ta,
)
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import (
    geometry_attributes as geom_attr,
    grid_manager as gm,
    vertical as v_grid,
)
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.interpolation import interpolation_attributes as intp_attr
from icon4py.model.common.io import io as common_io
from icon4py.model.common.metrics import metrics_attributes as metrics_attr
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    static_fields,
)
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.standalone_driver import (
    config as driver_config,
    driver_constants,
    driver_io,
    driver_states,
    driver_utils,
)


log = logging.getLogger(__name__)


class Icon4pyDriver:
    def __init__(
        self,
        *,
        config: driver_config.ExperimentConfig,
        backend: gtx.typing.Backend | None,
        grid: IconGrid,
        decomposition_info: decomposition_defs.DecompositionInfo,
        static_field_factories: static_fields.StaticFieldFactories,
        granules: driver_utils.Granules,
        vertical_grid_config: v_grid.VerticalGridConfig,
        exchange: decomposition_defs.ExchangeRuntime,
        global_reductions: decomposition_defs.Reductions,
        io_monitor: common_io.IOMonitor | None = None,
    ):
        self.config = config
        self.io_monitor = io_monitor
        self.backend = backend
        self.grid = grid
        self.decomposition_info = decomposition_info
        self.static_field_factories = static_field_factories
        self.granules = granules
        self.vertical_grid_config = vertical_grid_config
        self.model_time_variables = driver_states.ModelTimeVariables(config=config.driver)
        self.timer_collection = driver_states.TimerCollection(
            [timer.value for timer in driver_states.DriverTimers]
        )
        self.exchange = exchange
        self.global_reductions = global_reductions

        driver_utils.display_driver_setup_in_log_file(
            config=self.config.driver,
            model_time_variables=self.model_time_variables,
            vertical_params=self.static_field_factories.metrics._vertical_grid,
            tracer_config=self.config.tracer_config,
        )

    @functools.cached_property
    def _allocator(self) -> gtx.typing.Backend:
        return model_backends.get_allocator(self.backend)

    @functools.cached_property
    def _xp(self) -> types.ModuleType:
        return data_alloc.import_array_ns(self._allocator)

    def _is_last_substep(self, step_nr: int) -> bool:
        return step_nr == (self.model_time_variables.ndyn_substeps_var - 1)

    @staticmethod
    def _is_first_substep(step_nr: int) -> bool:
        return step_nr == 0

    def _full_name(self, func: Callable) -> str:
        return f"{self.__class__.__name__}:{func.__name__}"

    @functools.cached_property
    def _diagnostics_computer(self) -> driver_io.DiagnosticsComputer:
        """Reuses its scratch/output buffers across output steps (allocated once)."""
        return driver_io.DiagnosticsComputer(grid=self.grid, backend=self.backend)

    def _store_output(
        self,
        prognostic_state: prognostics.PrognosticState,
        simulation_current_datetime: time.AbsoluteTime,
    ) -> None:
        """Assemble the prognostic + diagnostic fields and hand them to the IO monitor.

        The assembled DataArrays reference the live state (see ``io.utils.to_data_array``),
        so they must be written here and now -- before the next step mutates the state. The
        static diagnostic inputs are fetched directly from the field factories.
        """
        assert self.io_monitor is not None
        metrics = self.static_field_factories.metrics
        interpolation = self.static_field_factories.interpolation
        state_to_store = driver_io.prognostic_state_to_dataarrays(prognostic_state)
        diagnostic_fields = self._diagnostics_computer.compute(
            prognostic_state,
            ddqz_z_full=metrics.get(metrics_attr.DDQZ_Z_FULL),
            rbf_vec_coeff_c1=interpolation.get(intp_attr.RBF_VEC_COEFF_C1),
            rbf_vec_coeff_c2=interpolation.get(intp_attr.RBF_VEC_COEFF_C2),
        )
        state_to_store.update(driver_io.diagnostic_fields_to_dataarrays(diagnostic_fields))
        self.io_monitor.store(state_to_store, simulation_current_datetime)

    def time_integration(
        self,
        ds: driver_states.DriverStates,
        do_prep_adv: bool,
    ) -> None:
        diffusion_diagnostic_state = ds.diffusion_diagnostic
        solve_nonhydro_diagnostic_state = ds.solve_nonhydro_diagnostic
        tracer_advection_diagnostic_state = ds.tracer_advection_diagnostic
        prognostic_states = ds.prognostics
        prep_adv = ds.prep_advection_prognostic
        tracer_prep_adv = ds.prep_tracer_advection_prognostic

        log.debug(
            f"starting time loop for dtime = {self.model_time_variables.dtime_in_seconds} s, substep_timestep = {self.model_time_variables.substep_timestep} s, n_timesteps = {self.model_time_variables.n_time_steps}"
        )

        # TODO(OngChia): Initialize vn tendencies that are used in solve_nh and advection to zero (init_ddt_vn_diagnostics subroutine)

        wall_clock_starting_time = datetime.datetime.now()

        try:  # fail gracefully and close `io_monitor` if something goes wrong
            if self.io_monitor is not None:
                # write the initial state; the simulation datetime is still the start here
                # (it is advanced below, per step)
                self._store_output(
                    prognostic_states.current, self.model_time_variables.simulation_current_datetime
                )

            for time_step in range(self.model_time_variables.n_time_steps):
                if self.config.driver.profiling_stats is not None:
                    if not self.config.driver.profiling_stats.skip_first_timestep or time_step > 0:
                        gtx_config.COLLECT_METRICS_LEVEL = (
                            self.config.driver.profiling_stats.gt4py_metrics_level
                        )

                log.info(
                    f"\n"
                    f"simulation date : {self.model_time_variables.simulation_current_datetime}, at timestep : {time_step}, Elapsed wall clock time: {(datetime.datetime.now() - wall_clock_starting_time).total_seconds()}"
                    f"\n"
                )

                self.model_time_variables.advance_simulation_datetime()

                self._integrate_one_time_step(
                    diffusion_diagnostic_state=diffusion_diagnostic_state,
                    solve_nonhydro_diagnostic_state=solve_nonhydro_diagnostic_state,
                    tracer_advection_diagnostic_state=tracer_advection_diagnostic_state,
                    prognostic_states=prognostic_states,
                    prep_adv=prep_adv,
                    do_prep_adv=do_prep_adv,
                    tracer_prep_adv=tracer_prep_adv,
                )
                device_utils.sync(self.backend)

                self.model_time_variables.is_first_step_in_simulation = False

                if self.config.nonhydrostatic is not None:
                    assert solve_nonhydro_diagnostic_state is not None
                    self._adjust_ndyn_substeps_var(solve_nonhydro_diagnostic_state)

                if self.io_monitor is not None:
                    self._store_output(
                        prognostic_states.current,
                        self.model_time_variables.simulation_current_datetime,
                    )
        finally:
            if self.io_monitor is not None:
                self.io_monitor.close()

        self._compute_mean_at_final_time_step(prognostic_states.current)

        self.timer_collection.show_timer_report()
        if (
            self.config.driver.profiling_stats is not None
            and self.config.driver.profiling_stats.gt4py_metrics_level > gtx_metrics.DISABLED
        ):
            print(gtx_metrics.dumps())
            gtx_metrics.dump_json(self.config.driver.profiling_stats.gt4py_metrics_output_file)

    def _integrate_one_time_step(
        self,
        *,
        diffusion_diagnostic_state: diffusion_states.DiffusionDiagnosticState | None,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro | None,
        tracer_advection_diagnostic_state: advection_states.AdvectionDiagnosticState | None,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection | None,
        do_prep_adv: bool,
        tracer_prep_adv: advection_states.AdvectionPrepAdvState | None,
    ) -> None:
        if self.config.nonhydrostatic is not None:
            assert solve_nonhydro_diagnostic_state is not None
            assert prep_adv is not None
            log.debug(f"Running {self.granules.solve_nonhydro.__class__}")
            self._do_dyn_substepping(
                solve_nonhydro_diagnostic_state,
                prognostic_states,
                prep_adv,
                do_prep_adv,
            )

        if self.granules.diffusion is not None:
            assert diffusion_diagnostic_state is not None
            if self.granules.diffusion.config.apply_to_horizontal_wind:
                log.debug(f"Running {self.granules.diffusion.__class__}")
                timer_diffusion = (
                    self.timer_collection.timers[
                        driver_states.DriverTimers.DIFFUSION_FIRST_STEP.value
                    ]
                    if self.model_time_variables.is_first_step_in_simulation
                    else self.timer_collection.timers[driver_states.DriverTimers.DIFFUSION.value]
                )
                with timer_diffusion:
                    self.granules.diffusion.run(
                        diffusion_diagnostic_state,
                        prognostic_states.next,
                        self.model_time_variables.dtime_in_seconds,
                    )

        # TODO(ricoh): [c34] optionally move the loop into the granule (for efficiency gains)
        # Precondition: passing data test with ntracer > 0
        if self.granules.tracer_advection is not None:
            assert tracer_advection_diagnostic_state is not None
            assert tracer_prep_adv is not None
            for tracer_current in prognostic_states.current.tracer.active_fields():
                tracer_next_field = getattr(prognostic_states.next.tracer, tracer_current.name)
                assert tracer_next_field is not None, (
                    f"tracer '{tracer_current.name}' active in current state but missing in next state"
                )
                self.granules.tracer_advection.run(
                    diagnostic_state=tracer_advection_diagnostic_state,
                    prep_adv=tracer_prep_adv,
                    p_tracer_now=tracer_current.field,
                    p_tracer_new=tracer_next_field,
                    dtime=self.model_time_variables.dtime_in_seconds,
                )

        prognostic_states.swap()

    def _update_time_levels_for_velocity_tendencies(
        self,
        diagnostic_state_nh: dycore_states.DiagnosticStateNonHydro,
        at_first_substep: bool,
        at_initial_timestep: bool,
    ) -> None:
        """
        Set time levels of advective tendency fields for call to velocity_tendencies.

        When using `TimeSteppingScheme.MOST_EFFICIENT` (itime_scheme=4 in ICON Fortran),
        `vertical_wind_advective_tendency.predictor` (advection term in vertical momentum equation in
        predictor step) is not computed in the predictor step of each substep.
        Instead, the advection term computed in the corrector step during the
        previous substep is reused for efficiency (except, of course, in the
        very first substep of the initial time step).
        `normal_wind_advective_tendency.predictor` (advection term in horizontal momentum equation in
        predictor step) is only computed in the predictor step of the first
        substep and the advection term in the corrector step during the previous
        substep is reused for `normal_wind_advective_tendency.predictor` from the second substep onwards.
        Additionally, in this scheme the predictor and corrector outputs are kept
        in separate elements of the pair (.predictor for the predictor step and
        .corrector for the corrector step) and interpoolated at the end of the
        corrector step to get the final output.

        No other time stepping schemes are currently supported.

        Args:
            diagnostic_state_nh: Diagnostic fields calculated in the dynamical core (SolveNonHydro)
            at_first_substep: Flag indicating if this is the first substep of the time step.
            at_initial_timestep: Flag indicating if this is the first time step.

        Returns:
            The index of the pair element to be used for the corrector output.
        """
        if not (at_initial_timestep and at_first_substep):
            diagnostic_state_nh.vertical_wind_advective_tendency.swap()
        if not at_first_substep:
            diagnostic_state_nh.normal_wind_advective_tendency.swap()

    def _do_dyn_substepping(
        self,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        do_prep_adv: bool,
    ) -> None:
        # TODO(OngChia): compute airmass for prognostic_state here

        timer_solve_nh = (
            self.timer_collection.timers[driver_states.DriverTimers.SOLVE_NH_FIRST_STEP.value]
            if self.model_time_variables.is_first_step_in_simulation
            else self.timer_collection.timers[driver_states.DriverTimers.SOLVE_NH.value]
        )
        for dyn_substep in range(self.model_time_variables.ndyn_substeps_var):
            self._compute_statistics(dyn_substep, prognostic_states.current)

            self._update_time_levels_for_velocity_tendencies(
                solve_nonhydro_diagnostic_state,
                at_first_substep=self._is_first_substep(dyn_substep),
                at_initial_timestep=self.model_time_variables.is_first_step_in_simulation,
            )

            with timer_solve_nh:
                assert self.granules.solve_nonhydro is not None
                self.granules.solve_nonhydro.time_step(
                    diagnostic_state_nh=solve_nonhydro_diagnostic_state,
                    prognostic_states=prognostic_states,
                    prep_adv=prep_adv,
                    second_order_divdamp_factor=self._update_spinup_second_order_divergence_damping(),
                    dtime=self.model_time_variables.substep_timestep,
                    ndyn_substeps_var=self.model_time_variables.ndyn_substeps_var,
                    at_initial_timestep=self.model_time_variables.is_first_step_in_simulation,
                    lprep_adv=do_prep_adv,
                    at_first_substep=self._is_first_substep(dyn_substep),
                    at_last_substep=self._is_last_substep(dyn_substep),
                )

            if not self._is_last_substep(dyn_substep):
                prognostic_states.swap()
        self._compute_total_mass_and_energy(prognostic_states.next)

        # TODO(OngChia): compute airmass for prognostic_state here

    # watch_mode is true if step is <= 1 or cfl already near or exceeding threshold.
    # omit spinup feature and the option that if the model starts from IFS or COSMO data
    # horizontal cfl is not ported
    def _adjust_ndyn_substeps_var(
        self,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
    ) -> None:
        global_max_vertical_cfl = self.global_reductions.max(
            self._xp.asarray(
                solve_nonhydro_diagnostic_state.max_vertical_cfl[()], dtype=ta.wpfloat
            ),
        )
        if (
            global_max_vertical_cfl
            > driver_constants.CFL_ENTER_WATCHMODE_FACTOR
            * self.config.driver.vertical_cfl_threshold
            and not self.model_time_variables.cfl_watch_mode
        ):
            log.warning(
                "High CFL number for vertical advection in dynamical core, entering watch mode"
            )
            self.model_time_variables.update_cfl_watch_mode(True)

        if self.model_time_variables.cfl_watch_mode:
            substep_fraction = ta.wpfloat(
                self.model_time_variables.ndyn_substeps_var / self.config.driver.ndyn_substeps
            )
            if (
                global_max_vertical_cfl * substep_fraction
                > driver_constants.CFL_THRESHOLD_FACTOR * self.config.driver.vertical_cfl_threshold
            ):
                log.warning(
                    f"Maximum vertical CFL number {global_max_vertical_cfl} is close to critical threshold"
                )

            vertical_cfl_threshold_for_increment = self.config.driver.vertical_cfl_threshold
            vertical_cfl_threshold_for_decrement = (
                driver_constants.CFL_THRESHOLD_FACTOR * self.config.driver.vertical_cfl_threshold
            )

            if global_max_vertical_cfl > vertical_cfl_threshold_for_increment:
                if self._xp.isfinite(global_max_vertical_cfl):
                    ndyn_substeps_increment = max(
                        1,
                        round(
                            self.model_time_variables.ndyn_substeps_var
                            * (global_max_vertical_cfl - vertical_cfl_threshold_for_increment)
                            / vertical_cfl_threshold_for_increment
                        ),
                    )
                    new_ndyn_substeps_var = min(
                        self.model_time_variables.ndyn_substeps_var + ndyn_substeps_increment,
                        self.model_time_variables.max_ndyn_substeps,
                    )
                else:
                    log.warning(
                        f"WARNING: max cfl {global_max_vertical_cfl} is not a number! Number of substeps is set to the max value! "
                    )
                    new_ndyn_substeps_var = self.model_time_variables.max_ndyn_substeps
                self.model_time_variables.update_ndyn_substeps(new_ndyn_substeps_var)
                # TODO (Chia Rui): check if we need to set ndyn_substeps_var in advection_config as in ICON when tracer advection is implemented
                log.warning(
                    f"The number of dynamics substeps is increased to {self.model_time_variables.ndyn_substeps_var}"
                )
            if (
                self.model_time_variables.ndyn_substeps_var > self.config.driver.ndyn_substeps
                and global_max_vertical_cfl
                * ta.wpfloat(
                    self.model_time_variables.ndyn_substeps_var
                    / (self.model_time_variables.ndyn_substeps_var - 1)
                )
                < vertical_cfl_threshold_for_decrement
            ):
                self.model_time_variables.update_ndyn_substeps(
                    self.model_time_variables.ndyn_substeps_var - 1
                )
                # TODO (Chia Rui): check if we need to set ndyn_substeps_var in advection_config as in ICON when tracer advection is implemented
                log.warning(
                    f"The number of dynamics substeps is decreased to {self.model_time_variables.ndyn_substeps_var}"
                )

                if (
                    self.model_time_variables.ndyn_substeps_var == self.config.driver.ndyn_substeps
                    and global_max_vertical_cfl
                    < driver_constants.CFL_LEAVE_WATCHMODE_FACTOR
                    * self.config.driver.vertical_cfl_threshold
                ):
                    log.warning(
                        "CFL number for vertical advection in dynamical core has decreased, leaving watch mode"
                    )
                    self.model_time_variables.update_cfl_watch_mode(False)

        # reset max_vertical_cfl to zero
        solve_nonhydro_diagnostic_state.max_vertical_cfl = data_alloc.scalar_like_array(
            0.0, self._allocator
        )

    def _update_spinup_second_order_divergence_damping(self) -> ta.wpfloat:
        if self.config.driver.apply_extra_second_order_divdamp:
            assert self.config.nonhydrostatic is not None
            fourth_order_divdamp_factor = self.config.nonhydrostatic.fourth_order_divdamp_factor
            if (
                self.model_time_variables.elapsed_time_in_seconds
                <= driver_constants.INITIAL_PERIOD_FOR_SECOND_ORDER_DIVDAMP
            ):
                return (
                    driver_constants.ADJUST_FACTOR_FOR_SECOND_ORDER_DIVDAMP
                    * fourth_order_divdamp_factor
                )
            elif (
                self.model_time_variables.elapsed_time_in_seconds
                <= driver_constants.TRANSITION_END_PERIOD_FOR_SECOND_ORDER_DIVDAMP
            ):
                return (
                    driver_constants.ADJUST_FACTOR_FOR_SECOND_ORDER_DIVDAMP
                    * fourth_order_divdamp_factor
                    * (
                        self.model_time_variables.elapsed_time_in_seconds
                        - driver_constants.INITIAL_PERIOD_FOR_SECOND_ORDER_DIVDAMP
                    )
                    / (
                        driver_constants.TRANSITION_END_PERIOD_FOR_SECOND_ORDER_DIVDAMP
                        - driver_constants.INITIAL_PERIOD_FOR_SECOND_ORDER_DIVDAMP
                    )
                )
            else:
                return ta.wpfloat("0.0")
        else:
            return ta.wpfloat("0.0")

    def _compute_statistics(
        self, current_dyn_substep: int, prognostic_states: prognostics.PrognosticState
    ) -> None:
        """
        Compute relevant statistics of prognostic variables at the beginning of every time step. The statistics include:
        absolute maximum value of rho, vn, and w, as well as the levels at which their maximum value is found.
        """
        if self.config.driver.enable_statistics_output:
            # TODO (Chia Rui): Do global max when multinode is ready
            rho_arg_max, max_rho = driver_utils.find_maximum_from_field(
                prognostic_states.rho,
            )
            vn_arg_max, max_vn = driver_utils.find_maximum_from_field(
                prognostic_states.vn,
            )
            w_arg_max, max_w = driver_utils.find_maximum_from_field(prognostic_states.w)

            def _determine_sign(input_number: float) -> str:
                return " " if input_number >= 0.0 else "-"

            rho_sign = _determine_sign(max_rho)
            vn_sign = _determine_sign(max_vn)
            w_sign = _determine_sign(max_w)

            log.info(
                f"substep / n_substeps : {current_dyn_substep:3d} / {self.model_time_variables.ndyn_substeps_var:3d} == "
                f"MAX RHO: {rho_sign}{abs(max_rho):.5e} at lvl {rho_arg_max[1]:4d}, MAX VN: {vn_sign}{abs(max_vn):.5e} at lvl {vn_arg_max[1]:4d}, MAX W: {w_sign}{abs(max_w):.5e} at lvl {w_arg_max[1]:4d}"
            )
        else:
            log.info(
                f"substep / n_substeps : {current_dyn_substep:3d} / {self.model_time_variables.ndyn_substeps_var:3d}"
            )

    def _compute_total_mass_and_energy(
        self, prognostic_states: prognostics.PrognosticState
    ) -> None:
        if self.config.driver.enable_statistics_output:
            rho_ndarray = prognostic_states.rho.ndarray
            cell_area_ndarray = self.static_field_factories.geometry.get(
                geom_attr.CELL_AREA
            ).ndarray
            cell_thickness_ndarray = self.static_field_factories.metrics.get(
                metrics_attr.DDQZ_Z_FULL
            ).ndarray
            local_mass = (
                rho_ndarray * cell_area_ndarray[:, self._xp.newaxis] * cell_thickness_ndarray
            )
            global_total_mass = self.global_reductions.sum(local_mass)
            # TODO (Chia Rui): compute total energy
            log.info(f"GLOBAL TOTAL MASS: {global_total_mass:.15e} kg")

    def _compute_mean_at_final_time_step(
        self, prognostic_states: prognostics.PrognosticState
    ) -> None:
        if self.config.driver.enable_statistics_output:
            rho_ndarray = prognostic_states.rho.ndarray
            vn_ndarray = prognostic_states.vn.ndarray
            w_ndarray = prognostic_states.w.ndarray
            theta_v_ndarray = prognostic_states.theta_v.ndarray
            exner_ndarray = prognostic_states.exner.ndarray
            log.info("")
            log.info("Global mean of    rho         vn           w          theta_v     exner:")
            log.info(
                f"{self.global_reductions.mean(rho_ndarray):.5e} "
                f"{self.global_reductions.mean(vn_ndarray):.5e} "
                f"{self.global_reductions.mean(w_ndarray):.5e} "
                f"{self.global_reductions.mean(theta_v_ndarray):.5e} "
                f"{self.global_reductions.mean(exner_ndarray):.5e} "
            )


def initialize_driver(
    *,
    config: driver_config.ExperimentConfig,
    grid_manager: gm.GridManager,
    process_props: decomposition_defs.ProcessProperties,
    backend: gtx.typing.Backend | None,
) -> Icon4pyDriver:
    output_path = driver_config.prepare_output_directory(
        config_output_path=config.driver.output_path,
        cli_output_path=None,
        process_props=process_props,
    )
    config = dataclasses.replace(
        config, driver=dataclasses.replace(config.driver, output_path=output_path)
    )

    allocator = model_backends.get_allocator(backend)

    decomposition_info = grid_manager.decomposition_info
    exchange = decomposition_defs.create_exchange(process_props, decomposition_info)
    global_reductions = decomposition_defs.create_reduction(process_props, decomposition_info)

    log.info("initializing the vertical grid")
    vertical_grid = driver_utils.create_vertical_grid(
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
    )

    log.info("initializing the topography")
    cell_topography = topography.create(
        config=config.topography,
        grid_manager=grid_manager,
        backend=backend,
        exchange=exchange,
    )

    log.info("initializing the static-field factories")
    static_field_factories = driver_utils.create_static_field_factories(
        grid_manager=grid_manager,
        decomposition_info=decomposition_info,
        vertical_grid=vertical_grid,
        cell_topography=gtx.as_field((dims.CellDim,), data=cell_topography, allocator=allocator),  # type: ignore[arg-type] # due to array_ns opacity
        backend=backend,
        process_props=process_props,
        exchange=exchange,
        global_reductions=global_reductions,
        geometry_config=config.geometry,
        interpolation_config=config.interpolation,
        metrics_config=config.metrics,
    )

    log.info("initializing granules")
    granules = driver_utils.initialize_granules(
        config=config,
        grid=grid_manager.grid,
        vertical_grid=vertical_grid,
        static_field_factories=static_field_factories,
        exchange=exchange,
        owner_mask=gtx.as_field(
            (dims.CellDim,),
            decomposition_info.owner_mask(dims.CellDim),  # type: ignore[arg-type]  # due to array_ns opacity
            allocator=allocator,
        ),
        backend=backend,
    )
    io_monitor = None
    if config.driver.enable_output:
        if process_props.comm_size > 1:
            # IO is single-node only for now: under MPI every rank would construct its own
            # monitor and write overlapping files. Disable until IO becomes distributed.
            log.warning("output is not supported in distributed (MPI) runs yet: disabling IO")
        else:
            log.info("Initializing single-node IO monitor")
            io_monitor = driver_io.create_io_monitor(
                output_path=config.driver.output_path,
                grid_file_path=pathlib.Path(grid_manager.file_path),
                grid=grid_manager.grid,
                vertical_grid=vertical_grid,
                dtime=config.driver.dtime,
                process_props=process_props,
            )

    icon4py_driver = Icon4pyDriver(
        config=config,
        backend=backend,
        grid=grid_manager.grid,
        decomposition_info=decomposition_info,
        static_field_factories=static_field_factories,
        granules=granules,
        vertical_grid_config=config.vertical_grid,
        exchange=exchange,
        global_reductions=global_reductions,
        io_monitor=io_monitor,
    )

    return icon4py_driver


def run_driver(
    *,
    config: driver_config.ExperimentConfig,
    grid_manager: gm.GridManager,
    process_props: decomposition_defs.ProcessProperties,
    backend: gtx.typing.Backend | None,
) -> tuple[driver_states.DriverStates, Icon4pyDriver]:
    icon4py_driver = initialize_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )
    allocator = model_backends.get_allocator(backend)
    prognostic_state_now = prognostics.initialize_prognostic_state(
        grid=icon4py_driver.grid,
        allocator=allocator,
        tracer_config=icon4py_driver.config.tracer_config,
    )
    initial_condition.create(
        config=icon4py_driver.config.initial_condition,
        vertical_config=icon4py_driver.config.vertical_grid,
        grid=icon4py_driver.grid,
        static_fields=icon4py_driver.static_field_factories,
        prognostic_state_now=prognostic_state_now,
        backend=icon4py_driver.backend,
        exchange=icon4py_driver.exchange,
        global_reductions=icon4py_driver.global_reductions,
    )
    diagnostic_state = diagnostics.initialize_diagnostic_state(
        grid=icon4py_driver.grid, allocator=allocator
    )
    ds = driver_states.assemble_driver_states(
        grid=icon4py_driver.grid,
        allocator=allocator,
        backend=icon4py_driver.backend,
        exchange=icon4py_driver.exchange,
        static_fields=icon4py_driver.static_field_factories,
        prognostic_state_now=prognostic_state_now,
        diagnostic_state=diagnostic_state,
        experiment_config=icon4py_driver.config,
    )
    driver_utils.validate_granule_state_consistency(
        config=icon4py_driver.config,
        granules=icon4py_driver.granules,
        states=ds,
    )
    icon4py_driver.time_integration(ds, do_prep_adv=False)
    return ds, icon4py_driver

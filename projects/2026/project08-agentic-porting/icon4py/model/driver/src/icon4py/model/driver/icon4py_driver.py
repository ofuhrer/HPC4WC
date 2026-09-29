# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import logging
import pathlib
from collections.abc import Callable
from typing import NamedTuple

import click
import numpy as np
from devtools import Timer
from gt4py.next import config as gtx_config, typing as gtx_typing
from gt4py.next.instrumentation import metrics as gtx_metrics

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import model_backends, model_options
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.common.utils import device_utils
from icon4py.model.driver import (
    icon4py_configuration as driver_config,
    initialization_utils as driver_init,
)


log = logging.getLogger(__name__)


class TimeLoop:
    @classmethod
    def name(cls):
        return cls.__name__

    def __init__(
        self,
        run_config: driver_config.Icon4pyRunConfig,
        diffusion_granule: diffusion.Diffusion,
        solve_nonhydro_granule: solve_nh.SolveNonhydro,
    ):
        self.run_config: driver_config.Icon4pyRunConfig = run_config
        self.diffusion = diffusion_granule
        self.solve_nonhydro = solve_nonhydro_granule

        self._n_time_steps: int = int(
            (self.run_config.end_date - self.run_config.start_date) / self.run_config.dtime
        )
        self.dtime_in_seconds: float = self.run_config.dtime.total_seconds()
        self._n_substeps_var: int = self.run_config.n_substeps
        self._substep_timestep: float = float(self.dtime_in_seconds / self._n_substeps_var)

        self._validate_config()

        # current simulation date
        self._simulation_date: datetime.datetime = self.run_config.start_date

        self._is_first_step_in_simulation: bool = not self.run_config.restart_mode
        self._allocator: gtx_typing.Allocator = model_backends.get_allocator(
            self.run_config.backend
        )

    def re_init(self):
        self._simulation_date = self.run_config.start_date
        self._is_first_step_in_simulation = True
        self._n_substeps_var = self.run_config.n_substeps

    def _validate_config(self):
        if self._n_time_steps < 0:
            raise ValueError("end_date should be larger than start_date. Please check.")

    @property
    def first_step_in_simulation(self):
        return self._is_first_step_in_simulation

    def _is_last_substep(self, step_nr: int):
        return step_nr == (self.n_substeps_var - 1)

    @staticmethod
    def _is_first_substep(step_nr: int):
        return step_nr == 0

    def _next_simulation_date(self):
        self._simulation_date += self.run_config.dtime

    @property
    def n_substeps_var(self):
        return self._n_substeps_var

    @property
    def simulation_date(self):
        return self._simulation_date

    @property
    def n_time_steps(self):
        return self._n_time_steps

    @property
    def substep_timestep(self):
        return self._substep_timestep

    def _full_name(self, func: Callable):
        return f"{self.__class__.__name__}:{func.__name__}"

    def time_integration(
        self,
        *,
        diffusion_diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        second_order_divdamp_factor: float,
        do_prep_adv: bool,
        profiling: driver_config.ProfilingConfig | None = None,
    ):
        log.info(
            f"starting time loop for dtime={self.dtime_in_seconds} s and n_timesteps={self._n_time_steps}"
        )
        log.info(
            f"apply_to_horizontal_wind={self.diffusion.config.apply_to_horizontal_wind} initial_stabilization={self.run_config.apply_initial_stabilization} dtime={self.dtime_in_seconds} s, substep_timestep={self._substep_timestep}"
        )

        # TODO(OngChia): Initialize vn tendencies that are used in solve_nh and advection to zero (init_ddt_vn_diagnostics subroutine)

        # TODO(OngChia): Compute diagnostic variables: P, T, zonal and meridonial winds, necessary for JW test output (diag_for_output_dyn subroutine)

        # TODO(OngChia): Initialize exner_pr used in solve_nh (compute_exner_pert subroutine)

        if (
            self.diffusion.config.apply_to_horizontal_wind
            and self.run_config.apply_initial_stabilization
            and self._is_first_step_in_simulation
        ):
            log.info("running initial step to diffuse fields before time loop starts")
            self.diffusion.run(
                diffusion_diagnostic_state,
                prognostic_states.current,
                self.dtime_in_seconds,
                initial_run=True,
            )
        log.info(
            f"starting real time loop for dtime={self.dtime_in_seconds} n_timesteps={self._n_time_steps}"
        )
        timer_first_timestep = Timer("TimeLoop: first time step", dp=6)
        timer_after_first_timestep = Timer("TimeLoop: after first time step", dp=6)
        for time_step in range(self._n_time_steps):
            timer = timer_first_timestep if time_step == 0 else timer_after_first_timestep
            if profiling is not None:
                if not profiling.skip_first_timestep or time_step > 0:
                    gtx_config.COLLECT_METRICS_LEVEL = profiling.gt4py_metrics_level

            log.info(f"simulation date : {self._simulation_date} run timestep : {time_step}")
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f" MAX VN: {np.abs(prognostic_states.current.vn.asnumpy()).max():.15e} , MAX W: {np.abs(prognostic_states.current.w.asnumpy()).max():.15e}"
                )
                log.debug(
                    f" MAX RHO: {np.abs(prognostic_states.current.rho.asnumpy()).max():.15e} , MAX THETA_V: {np.abs(prognostic_states.current.theta_v.asnumpy()).max():.15e}"
                )
                # TODO(OngChia): check with Anurag about printing of max and min of variables. Currently, these max values are only output at debug level. There should be namelist parameters to control which variable max should be output.

            self._next_simulation_date()

            # update boundary conditions

            timer.start()
            self._integrate_one_time_step(
                diffusion_diagnostic_state=diffusion_diagnostic_state,
                solve_nonhydro_diagnostic_state=solve_nonhydro_diagnostic_state,
                prognostic_states=prognostic_states,
                prep_adv=prep_adv,
                second_order_divdamp_factor=second_order_divdamp_factor,
                do_prep_adv=do_prep_adv,
            )
            device_utils.sync(self._allocator)
            timer.capture()

            self._is_first_step_in_simulation = False

            # TODO(OngChia): modify n_substeps_var if cfl condition is not met. (set_dyn_substeps subroutine)

            # TODO(OngChia): compute diagnostic variables: P, T, zonal and meridonial winds, necessary for JW test output (diag_for_output_dyn subroutine)

            # TODO(OngChia): simple IO enough for JW test

        timer_first_timestep.summary(True)
        if self.n_time_steps > 1:  # in case only one time step was run
            timer_after_first_timestep.summary(True)
        if profiling is not None and profiling.gt4py_metrics_level > gtx_metrics.DISABLED:
            print(gtx_metrics.dumps())
            gtx_metrics.dump_json(profiling.gt4py_metrics_output_file)

    def _integrate_one_time_step(
        self,
        *,
        diffusion_diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        second_order_divdamp_factor: float,
        do_prep_adv: bool,
    ):
        # TODO(OngChia): Add update_spinup_damping here to compute second_order_divdamp_factor

        self._do_dyn_substepping(
            solve_nonhydro_diagnostic_state=solve_nonhydro_diagnostic_state,
            prognostic_states=prognostic_states,
            prep_adv=prep_adv,
            second_order_divdamp_factor=second_order_divdamp_factor,
            do_prep_adv=do_prep_adv,
        )

        if self.diffusion.config.apply_to_horizontal_wind:
            self.diffusion.run(
                diffusion_diagnostic_state,
                prognostic_states.next,
                self.dtime_in_seconds,
            )

        prognostic_states.swap()

    # TODO(OngChia): add tracer advection here

    def _update_time_levels_for_velocity_tendencies(
        self,
        diagnostic_state_nh: dycore_states.DiagnosticStateNonHydro,
        at_first_substep: bool,
        at_initial_timestep: bool,
    ):
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
        *,
        solve_nonhydro_diagnostic_state: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        second_order_divdamp_factor: float,
        do_prep_adv: bool,
    ):
        # TODO(OngChia): compute airmass for prognostic_state here

        for dyn_substep in range(self.n_substeps_var):
            log.info(
                f"simulation date : {self._simulation_date} substep / n_substeps : {dyn_substep} / "
                f"{self.n_substeps_var} , is_first_step_in_simulation : {self._is_first_step_in_simulation}"
            )

            self._update_time_levels_for_velocity_tendencies(
                solve_nonhydro_diagnostic_state,
                at_first_substep=self._is_first_substep(dyn_substep),
                at_initial_timestep=self._is_first_step_in_simulation,
            )

            self.solve_nonhydro.time_step(
                diagnostic_state_nh=solve_nonhydro_diagnostic_state,
                prognostic_states=prognostic_states,
                prep_adv=prep_adv,
                second_order_divdamp_factor=second_order_divdamp_factor,
                dtime=self._substep_timestep,
                ndyn_substeps_var=self.n_substeps_var,
                at_initial_timestep=self._is_first_step_in_simulation,
                lprep_adv=do_prep_adv,
                at_first_substep=self._is_first_substep(dyn_substep),
                at_last_substep=self._is_last_substep(dyn_substep),
            )

            if not self._is_last_substep(dyn_substep):
                prognostic_states.swap()

        # TODO(OngChia): compute airmass for prognostic_state here


class DriverStates(NamedTuple):
    """
    Initialized states for the driver run.

    Attributes:
        prep_advection_prognostic: Fields collecting data for advection during the solve nonhydro timestep.
        solve_nonhydro_diagnostic: Initial state for solve_nonhydro diagnostic variables.
        diffusion_diagnostic: Initial state for diffusion diagnostic variables.
        prognostics: Initial state for prognostic variables (double buffered).
        diagnostic: Initial state for global diagnostic variables.
    """

    prep_advection_prognostic: dycore_states.PrepAdvection
    solve_nonhydro_diagnostic: dycore_states.DiagnosticStateNonHydro
    diffusion_diagnostic: diffusion_states.DiffusionDiagnosticState
    prognostics: common_utils.TimeStepPair[prognostics.PrognosticState]
    diagnostic: diagnostics.DiagnosticState


class DriverParams(NamedTuple):
    """
    Parameters for the driver run.

    Attributes:
        second_order_divdamp_factor: Second order divergence damping factor.
    """

    second_order_divdamp_factor: float


def initialize(
    *,
    file_path: pathlib.Path,
    process_props: decomposition.ProcessProperties,
    serialization_type: driver_init.SerializationType,
    experiment_type: driver_init.ExperimentType,
    grid_file: pathlib.Path,
    backend_like: model_backends.BackendLike,
) -> tuple[TimeLoop, DriverStates, DriverParams]:
    """
    Initialize the driver run.

    This function does the following:
    - load configuration
    - load grid information
    - initialize components: diffusion and solve_nh
    - load diagnostic and prognostic variables (serialized data)
    - setup the time loop

    Parameters:
        file_path: Path to the serialized data.
        process_props: Process properties.
        serialization_type: Serialization type.
        experiment_type: Experiment type.
        grid_file: Path of the grid.
        backend: GT4Py backend.

    Returns:
        TimeLoop: Time loop object.
        DriverStates: Initial states for the driver run.
        DriverParams: Parameters for the driver run.
    """
    log.info("initialize parallel runtime")
    log.info(f"reading configuration: experiment {experiment_type}")
    config = driver_config.read_config(experiment_type=experiment_type, backend=backend_like)

    # TODO(havogt): Serialbox infrastructure needs to be upgraded to work with allocator instead of backend
    generic_concrete_backend = model_options.customize_backend(None, backend_like)

    decomp_info = driver_init.read_decomp_info(
        path=file_path,
        grid_file=grid_file,
        process_props=process_props,
        backend=generic_concrete_backend,
        ser_type=serialization_type,
    )

    log.info(f"initializing the grid from '{file_path}'")
    grid = driver_init.read_icon_grid(
        path=file_path,
        grid_file=grid_file,
        backend=generic_concrete_backend,
        rank=process_props.rank,
        ser_type=serialization_type,
    )
    log.info(f"reading input fields from '{file_path}'")
    (
        edge_geometry,
        cell_geometry,
        vertical_geometry,
        c_owner_mask,
    ) = driver_init.read_geometry_fields(
        path=file_path,
        grid_file=grid_file,
        vertical_grid_config=config.vertical_grid_config,
        backend=generic_concrete_backend,
        rank=process_props.rank,
        ser_type=serialization_type,
    )
    (
        diffusion_metric_state,
        diffusion_interpolation_state,
        solve_nonhydro_metric_state,
        solve_nonhydro_interpolation_state,
        _,
    ) = driver_init.read_static_fields(
        path=file_path,
        grid_file=grid_file,
        backend=generic_concrete_backend,
        rank=process_props.rank,
        ser_type=serialization_type,
    )

    log.info("initializing diffusion")
    diffusion_params = diffusion.DiffusionParams(config.diffusion_config)
    exchange = decomposition.create_exchange(process_props, decomp_info)
    diffusion_granule = diffusion.Diffusion(
        grid=grid,
        config=config.diffusion_config,
        params=diffusion_params,
        vertical_grid=vertical_geometry,
        metric_state=diffusion_metric_state,
        interpolation_state=diffusion_interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        exchange=exchange,
        backend=backend_like,
    )

    nonhydro_params = solve_nh.NonHydrostaticParams(config.solve_nonhydro_config)

    solve_nonhydro_granule = solve_nh.SolveNonhydro(
        grid=grid,
        backend=backend_like,
        config=config.solve_nonhydro_config,
        params=nonhydro_params,
        metric_state_nonhydro=solve_nonhydro_metric_state,
        interpolation_state=solve_nonhydro_interpolation_state,
        vertical_params=vertical_geometry,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=c_owner_mask,
        exchange=exchange,
    )

    (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        second_order_divdamp_factor,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    ) = driver_init.read_initial_state(
        grid=grid,
        cell_param=cell_geometry,
        edge_param=edge_geometry,
        path=file_path,
        backend=generic_concrete_backend,
        rank=process_props.rank,
        experiment_type=experiment_type,
    )
    prognostics_states = common_utils.TimeStepPair(prognostic_state_now, prognostic_state_next)

    time_loop = TimeLoop(
        run_config=config.run_config,
        diffusion_granule=diffusion_granule,
        solve_nonhydro_granule=solve_nonhydro_granule,
    )

    return (
        time_loop,
        DriverStates(
            prep_advection_prognostic=prep_adv,
            solve_nonhydro_diagnostic=solve_nonhydro_diagnostic_state,
            diffusion_diagnostic=diffusion_diagnostic_state,
            prognostics=prognostics_states,
            diagnostic=diagnostic_state,
        ),
        DriverParams(second_order_divdamp_factor=second_order_divdamp_factor),
    )


@click.command()
@click.argument("input_path")
@click.option(
    "--run_path",
    default="./",
    help="Folder for code logs to file and to console. Only debug logging is going to the file.",
)
@click.option(
    "--mpi",
    default=False,
    show_default=True,
    help="Whether or not you are running with mpi. Currently not fully tested yet.",
)
@click.option(
    "--serialization_type",
    default=driver_init.SerializationType.SB.value,
    show_default=True,
    help="Serialization type for grid info and static fields. This is currently the only possible way to load the grid info and static fields.",
)
@click.option(
    "--experiment_type",
    default=driver_init.ExperimentType.ANY.value,
    show_default=True,
    help="Option for configuration and how the initial state is generated. "
    "Setting it to the default value will instruct the model to use the default configuration of MeteoSwiss regional experiment and read the initial state from serialized data. "
    "Currently, users can also set it to either jabw or grauss_3d_torus to generate analytic initial condition for the JW and mountain wave tests, respectively (they are placed in abs_path_to_icon4py/model/driver/src/icon4py/model/driver/test_cases/).",
)
@click.option(
    "--grid_file",
    required=True,
    help="Path of the grid file.",
)
@click.option(
    "--enable_output",
    is_flag=True,
    default=False,
    help="Enable all debugging messages. Otherwise, only critical error messages are printed.",
)
@click.option(
    "--enable_profiling",
    is_flag=True,
    default=False,
    help="Enable detailed profiling with GT4Py metrics.",
)
@click.option(
    "--icon4py_driver_backend",
    "-b",
    required=True,
    help="Backend for all components executed in icon4py driver. For performance and stability, it is advised to choose between gtfn_cpu or gtfn_gpu. Please see abs_path_to_icon4py/model/common/src/icon4py/model/common/model_backends.py) ",
)
def icon4py_driver(
    *,
    input_path,
    run_path,
    mpi,
    serialization_type,
    experiment_type,
    grid_file,
    enable_output,
    enable_profiling,
    icon4py_driver_backend,
) -> None:
    """
    usage: python dycore_driver.py abs_path_to_icon4py/testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data

    Run the icon4py driver, where INPUT_PATH is the path to folder storing the serialized data.

    steps:
    1. initialize model from serialized data:

        a) load config of icon and components: diffusion and solve_nh

        b) initialize grid

        c) initialize/configure components ie "granules"

        d) load local, diagnostic and prognostic variables

        e) setup the time loop

    2. run time loop
    """

    if icon4py_driver_backend not in model_backends.BACKENDS:
        raise ValueError(
            f"Invalid driver backend: {icon4py_driver_backend}. \n"
            f"Available backends are {', '.join([f'{k}' for k in model_backends.BACKENDS])}"
        )
    backend_like = model_backends.BACKENDS[icon4py_driver_backend]

    process_props = decomposition.get_process_properties(decomposition.get_runtype(with_mpi=mpi))
    driver_init.configure_logging(run_path, experiment_type, enable_output, process_props)

    time_loop: TimeLoop
    ds: DriverStates
    dp: DriverParams
    time_loop, ds, dp = initialize(
        file_path=pathlib.Path(input_path),
        process_props=process_props,
        serialization_type=serialization_type,
        experiment_type=experiment_type,
        grid_file=pathlib.Path(grid_file),
        backend_like=backend_like,
    )
    log.info(f"Starting ICON dycore run: {time_loop.simulation_date.isoformat()}")
    log.info(
        f"input args: input_path={input_path}, n_time_steps={time_loop.n_time_steps}, ending date={time_loop.run_config.end_date}"
    )

    log.info(f"input args: input_path={input_path}, n_time_steps={time_loop.n_time_steps}")

    log.info("dycore configuring: DONE")
    log.info("time loop: START")

    time_loop.time_integration(
        diffusion_diagnostic_state=ds.diffusion_diagnostic,
        solve_nonhydro_diagnostic_state=ds.solve_nonhydro_diagnostic,
        prognostic_states=ds.prognostics,
        prep_adv=ds.prep_advection_prognostic,
        second_order_divdamp_factor=dp.second_order_divdamp_factor,
        do_prep_adv=False,
        profiling=driver_config.ProfilingConfig() if enable_profiling else None,
    )

    log.info("time loop:  DONE")


if __name__ == "__main__":
    icon4py_driver()

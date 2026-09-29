# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import pathlib
import re
from typing import Any

from gt4py.next.instrumentation import metrics as gtx_metrics

from icon4py.model.atmosphere.advection import advection as tracer_advection
from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    single_moment_six_class_gscp_graupel as graupel,
)
from icon4py.model.common import initial_condition, time, topography, type_alias as ta
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.grid.geometry_config import GeometryConfig
from icon4py.model.common.initial_condition import from_file
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states import tracer_state
from icon4py.model.common.utils import fortran_config


log = logging.getLogger(__name__)


@dataclasses.dataclass
class ProfilingStats:
    gt4py_metrics_level: int = gtx_metrics.ALL
    gt4py_metrics_output_file: str = "gt4py_metrics.json"
    skip_first_timestep: bool = True


# ISO 8601 duration, restricted to the fixed-length components (weeks, days,
# hours, minutes, seconds). Years and months are intentionally not matched since
# their length is not fixed, and this is currently only used for dtime.
_ISO8601_DURATION = re.compile(
    r"P(?:(?P<weeks>\d+)W)?(?:(?P<days>\d+)D)?"
    r"(?:T(?=\d)(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?"
)


def _timedelta_from_iso8601(duration: str) -> time.RelativeTime:
    """Parse an ISO 8601 duration such as 'PT300S' into a 'time.RelativeTime'.

    Only the components convertible to a fixed duration are supported (weeks,
    days, hours, minutes, seconds).
    """
    match = _ISO8601_DURATION.fullmatch(duration)
    if match is None or not any(match.groups()):
        raise ValueError(f"Invalid ISO 8601 duration: '{duration}'.")
    components = {name: float(value) for name, value in match.groupdict().items() if value}
    return time.RelativeTime(**components)


@dataclasses.dataclass(frozen=True)
class DriverConfig:
    """
    Standalone driver configuration.

    Default values should correspond to default values in ICON.
    """

    experiment_name: str
    profiling_stats: ProfilingStats | None
    dtime: time.RelativeTime
    start_of_simulation: time.AbsoluteTime
    end_of_simulation: time.EndOfSimulation
    output_path: pathlib.Path = dataclasses.field(default_factory=lambda: pathlib.Path("./output"))
    apply_extra_second_order_divdamp: bool = False
    vertical_cfl_threshold: ta.wpfloat = dataclasses.field(default_factory=lambda: ta.wpfloat(0.85))
    ndyn_substeps: int = 5
    enable_statistics_output: bool = False
    enable_output: bool = False

    @classmethod
    def from_fortran_dict(
        cls, *, atm_dict: dict[str, Any], master_dict: dict[str, Any], **overrides: Any
    ) -> DriverConfig:
        nonhydrostatic_nml = atm_dict["nonhydrostatic_nml"]
        run_nml = atm_dict["run_nml"]
        master_time_control_nml = master_dict["master_time_control_nml"]
        master_model_nml = master_dict["master_model_nml"]
        # Both 'modeltimestep' (an ISO 8601 duration) and 'dtime' (seconds) are
        # always present; a non-empty 'modeltimestep' takes priority over 'dtime'.
        modeltimestep = run_nml["modeltimestep"].strip()
        dtime = (
            _timedelta_from_iso8601(modeltimestep)
            if modeltimestep
            else time.RelativeTime(seconds=run_nml["dtime"])
        )
        start_datetime_str = master_time_control_nml["experimentstartdate"]
        end_datetime_str = master_time_control_nml["experimentstopdate"]
        return cls(
            experiment_name=master_model_nml["model_namelist_filename"]
            .removeprefix("NAMELIST_")
            .removesuffix("_sb_atm"),
            dtime=dtime,
            start_of_simulation=datetime.datetime.fromisoformat(
                start_datetime_str.replace("Z", "+00:00")
            ),
            end_of_simulation=datetime.datetime.fromisoformat(
                end_datetime_str.replace("Z", "+00:00")
            ),
            # apply_extra_second_order_divdamp does not have a namelist
            # variable in fortran. It is coded as follows in mo_nh_stepping.f90:
            # IF (elapsed_time_global <= 7200._wp+0.5_wp*dtime .AND. .NOT. ltestcase)
            apply_extra_second_order_divdamp=not run_nml.get("ltestcase", False),
            vertical_cfl_threshold=ta.wpfloat(str(nonhydrostatic_nml["vcfl_threshold"])),
            ndyn_substeps=nonhydrostatic_nml["ndyn_substeps"],
            **overrides,
        )


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    geometry: GeometryConfig
    metrics: metrics_factory.MetricsConfig
    interpolation: interpolation_factory.InterpolationConfig
    vertical_grid: v_grid.VerticalGridConfig
    topography: topography.TopographyConfig
    initial_condition: initial_condition.InitialConditionConfig
    driver: DriverConfig
    nonhydrostatic: solve_nh.NonHydrostaticConfig | None = None
    diffusion: diffusion.DiffusionConfig | None = None
    tracer_config: tracer_state.TracerConfig | None = None
    tracer_advection: tracer_advection.AdvectionConfig | None = None
    graupel: graupel.SingleMomentSixClassIconGraupelConfig | None = None

    def with_overrides(self, **overrides: Any) -> ExperimentConfig:
        replacements: dict[str, Any] = {}
        for key, value in overrides.items():
            current = getattr(self, key)
            if isinstance(value, dict):
                replacements[key] = dataclasses.replace(current, **value)
            else:
                replacements[key] = value
        return dataclasses.replace(self, **replacements)


def read_experiment_config_from_fortran(
    config_file_path: pathlib.Path,
    *,
    enable_profiling: bool = False,
    enable_statistics_output: bool = False,
) -> ExperimentConfig:
    """Assemble an :class:`ExperimentConfig` from a directory of serialized Fortran namelists."""

    with (config_file_path / fortran_config.ATM_DICT_FNAME).open() as f:
        atm_dict = json.load(f)
    with (config_file_path / fortran_config.MASTER_DICT_FNAME).open() as f:
        master_dict = json.load(f)
    with (config_file_path / fortran_config.INPUT_DICT_FNAME).open() as f:
        input_dict = json.load(f)

    geometry_cfg = GeometryConfig(use_analytical_means=True)

    metrics_cfg = metrics_factory.MetricsConfig.from_fortran_dict(atm_dict)

    interpolation_cfg = interpolation_factory.InterpolationConfig.from_fortran_dict(atm_dict)

    vertical_grid_cfg = v_grid.VerticalGridConfig.from_fortran_dict(atm_dict)

    topography_cfg = topography.TopographyConfig.from_fortran_dict(
        atm_dict=atm_dict, input_dict=input_dict, data_path=config_file_path
    )

    nonhydro_cfg = solve_nh.NonHydrostaticConfig.from_fortran_dict(
        atm_dict,
        max_nudging_coefficient=interpolation_cfg.max_nudging_coefficient,
    )

    diffusion_cfg = diffusion.DiffusionConfig.from_fortran_dict(
        atm_dict,
        max_nudging_coefficient=interpolation_cfg.max_nudging_coefficient,
    )

    do_tracer_advection = not (
        "exclaim_ch_r04b09_dsl" in config_file_path.name
        or "exclaim_ape_R02B04" in config_file_path.name
    )
    # The experiments above were run in fortran with a tracer advection scheme
    # that has not been ported to ICON4Py and can not be used for testing.
    # TODO (jcanton): this isn't the right place to keep a special case
    # handling. Either fix these experiments or move the special case handling.
    tracer_advection_cfg = (
        tracer_advection.AdvectionConfig.from_fortran_dict(atm_dict)
        if do_tracer_advection
        else None
    )
    ntracer = (
        fortran_config.list_to_value(atm_dict["run_nml"]["ntracer"]) if do_tracer_advection else 0
    )
    tracer_cfg = tracer_state.TracerConfig.from_ntracer(ntracer)

    do_physics = "nwp_phy_nml" in atm_dict and "nwp_tuning_nml" in atm_dict
    # If these two namelists are missing it means that the experiment was run
    # without microphysics and we have to skip parsing the graupel config which
    # relies on some of these parameters.
    graupel_cfg = (
        graupel.SingleMomentSixClassIconGraupelConfig.from_fortran_dict(atm_dict)
        if do_physics
        else None
    )

    initial_condition_cfg = initial_condition.InitialConditionConfig.from_fortran_dict(
        atm_dict=atm_dict, input_dict=input_dict, data_path=config_file_path
    )

    if not do_tracer_advection and isinstance(
        initial_condition_cfg.config, from_file.FromFileConfig
    ):
        initial_condition_cfg = dataclasses.replace(
            initial_condition_cfg,
            config=dataclasses.replace(initial_condition_cfg.config, ntracer=0),
        )

    profiling_stats = ProfilingStats() if enable_profiling else None
    driver_cfg = DriverConfig.from_fortran_dict(
        atm_dict=atm_dict,
        master_dict=master_dict,
        profiling_stats=profiling_stats,
        enable_statistics_output=enable_statistics_output,
    )

    return ExperimentConfig(
        geometry=geometry_cfg,
        metrics=metrics_cfg,
        interpolation=interpolation_cfg,
        vertical_grid=vertical_grid_cfg,
        topography=topography_cfg,
        nonhydrostatic=nonhydro_cfg,
        diffusion=diffusion_cfg,
        tracer_config=tracer_cfg,
        tracer_advection=tracer_advection_cfg,
        graupel=graupel_cfg,
        initial_condition=initial_condition_cfg,
        driver=driver_cfg,
    )


def prepare_output_directory(
    config_output_path: pathlib.Path,
    cli_output_path: pathlib.Path | None,
    process_props: Any | None = None,
) -> pathlib.Path:
    output_path = cli_output_path if cli_output_path is not None else config_output_path

    is_rank_zero = process_props is None or process_props.rank == 0

    if is_rank_zero:
        if output_path.exists():
            current_time = datetime.datetime.now()
            log.warning(f"output path {output_path} already exists, a time stamp will be added")
            output_path = (
                output_path.parent
                / f"{output_path.name}_{datetime.date.today()}_{current_time.hour}h_{current_time.minute}m_{current_time.second}s"
            )
        output_path.mkdir(parents=True, exist_ok=False)

    if process_props is not None and process_props.comm_size > 1:
        output_path = pathlib.Path(
            process_props.comm.bcast(str(output_path) if process_props.rank == 0 else None, root=0)
        )
        process_props.comm.Barrier()

    return output_path

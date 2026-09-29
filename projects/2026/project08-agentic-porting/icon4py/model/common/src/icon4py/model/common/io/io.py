# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import abc
import dataclasses
import datetime as dt
import logging
import pathlib
import uuid
from collections.abc import Sequence
from typing import Any, TypeAlias

from icon4py.model.common import exceptions, time
from icon4py.model.common.components import monitor
from icon4py.model.common.grid import base, vertical as v_grid
from icon4py.model.common.grid.vertical import VerticalGrid
from icon4py.model.common.io import cf_utils, ugrid, writers
from icon4py.model.common.io.writers import GlobalFileAttributes


log = logging.getLogger(__name__)


#: Output schedule given as either a number of model steps or a simulation-time delta.
#: A time delta is normalized to a number of steps internally (using the model time step),
#: so the schedule is always evaluated in steps.
OutputInterval: TypeAlias = time.RelativeTime | time.NumTimeSteps  # noqa: UP040


def _interval_in_steps(output_interval: OutputInterval, dtime: time.RelativeTime) -> int:
    """Normalize an output interval to a number of model steps."""
    if isinstance(output_interval, time.RelativeTime):
        steps = round(output_interval / dtime)
        if steps < 1:
            raise exceptions.InvalidConfigError(
                f"Output interval {output_interval} is shorter than the model time step {dtime}."
            )
        return steps
    return output_interval


class Config(abc.ABC):
    """
    Base class for all config classes.

    # TODO(halungge): Need to visit this, when we address configuration
    """

    def __str__(self) -> str:
        return f"instance of {self.__class__}(Config)"

    @abc.abstractmethod
    def validate(self) -> None:
        """
        Validate the config.

        Raises:
            InvalidConfigError: if the config is invalid
        """

        pass


@dataclasses.dataclass(frozen=True)
class FieldGroupIOConfig(Config):
    """
    Structured config for IO of a field group.

    Field group is a number of fields that are output at the same time intervals on the same grid
    (can be any horizontal dimension) and vertical levels.

    """

    filename: str
    variables: list[str]
    #: Output schedule: either a number of model steps (``int``) or a simulation-time
    #: delta (``datetime.timedelta``); a delta is normalized to steps using the model time
    #: step. Defaults to every step.
    output_interval: OutputInterval = time.NumTimeSteps(1)
    timesteps_per_file: int = 10
    nc_title: str = "ICON4Py Simulation"
    nc_comment: str = "ICON inspired code in Python and GT4Py"

    def __post_init__(self) -> None:
        self.validate()

    def _validate_filename(self) -> None:
        if not self.filename:
            raise exceptions.InvalidConfigError("Output filename is missing.")
        if self.filename.startswith("/"):
            raise exceptions.InvalidConfigError(
                f"Filename may not be an absolute path: {self.filename}."
            )

    def validate(self) -> None:
        # bool is a subclass of int, but is not a valid interval
        if isinstance(self.output_interval, bool) or not isinstance(
            self.output_interval, OutputInterval
        ):
            raise exceptions.InvalidConfigError(
                f"Output interval must be of type {OutputInterval}: {self.output_interval!r}."
            )
        positive = (
            self.output_interval > time.RelativeTime(0)
            if isinstance(self.output_interval, time.RelativeTime)
            else self.output_interval > 0
        )
        if not positive:
            raise exceptions.InvalidConfigError(
                f"Output interval must be positive: {self.output_interval!r}."
            )
        if not self.variables:
            raise exceptions.InvalidConfigError("No variables provided for output.")
        self._validate_filename()


@dataclasses.dataclass(frozen=True)
class IOConfig(Config):
    """
    Structured and hierarchical config for IO.

    Holds some general configuration and a collection of configurations for each field group.

    """

    output_path: str = "./output/"
    field_groups: Sequence[FieldGroupIOConfig] = ()

    time_units = cf_utils.DEFAULT_TIME_UNIT
    calendar = cf_utils.DEFAULT_CALENDAR

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.field_groups:
            log.warning("No field configurations provided for output")
        else:
            for field_config in self.field_groups:
                field_config.validate()


class IOMonitor(monitor.Monitor):
    """
    Composite Monitor for all IO groups.
    """

    def __init__(
        self,
        *,
        config: IOConfig,
        vertical_size: v_grid.VerticalGrid,
        horizontal_size: base.HorizontalGridSize,
        grid_file_name: pathlib.Path,
        grid_id: uuid.UUID,
        dtime: time.RelativeTime,
    ):
        self.config = config
        # ``grid_file_name`` is the source grid NetCDF, used solely to regenerate the UGRID
        # topology file (`_write_ugrid`); the grid identity comes from ``grid_id`` (the
        # ``Grid`` object), not from the file.
        # TODO(kotsaloscv): build the UGRID topology from ``Grid``/``GridGeometry`` so the
        # monitor no longer needs the source file path at all.
        self._grid_file = grid_file_name
        self._initialize_output()
        self._group_monitors = [
            FieldGroupMonitor(
                config=conf,
                vertical=vertical_size,
                horizontal=horizontal_size,
                grid_id=grid_id,
                output_path=self._output_path,
                dtime=dtime,
            )
            for conf in config.field_groups
        ]

    def _initialize_output(self) -> None:
        self._create_output_dir()
        self._write_ugrid()

    def _create_output_dir(self) -> None:
        path = pathlib.Path(self.config.output_path)
        # The directory may already exist: in the driver it is created upfront by
        # ``prepare_output_directory`` (which timestamps it if it already existed). Existing
        # *files* are kept safe though -- ``FieldGroupMonitor._init_dataset`` refuses to
        # overwrite an existing data file, so a rerun into a populated dir fails loudly.
        path.mkdir(parents=True, exist_ok=True)
        self._output_path = path

    def _write_ugrid(self) -> None:
        writer = ugrid.IconUGridWriter(self._grid_file, self._output_path)
        writer(validate=True)

    @property
    def path(self) -> pathlib.Path:
        return self._output_path

    def store(
        self, state: dict, model_time: dt.datetime, *args: Any, **kwargs: dict[str, Any]
    ) -> None:
        for m in self._group_monitors:
            m.store(state, model_time, *args, **kwargs)

    def close(self) -> None:
        for m in self._group_monitors:
            m.close()


class FieldGroupMonitor(monitor.Monitor):
    """
    Monitor for a group of fields.

    This monitor is responsible for storing a group of fields that are output at the same time intervals.
    """

    def __init__(
        self,
        *,
        config: FieldGroupIOConfig,
        vertical: VerticalGrid,
        horizontal: base.HorizontalGridSize,
        grid_id: uuid.UUID,
        dtime: time.RelativeTime,
        time_units: str = cf_utils.DEFAULT_TIME_UNIT,
        calendar: str = cf_utils.DEFAULT_CALENDAR,
        output_path: pathlib.Path = pathlib.Path(__file__).parent,
    ):
        self._global_attrs: GlobalFileAttributes = {
            "Conventions": "CF-1.7",  # TODO(halungge): check changelog? latest version is 1.11
            "title": config.nc_title,
            "comment": config.nc_comment,
            "institution": "ETH Zurich and MeteoSwiss",
            "source": "https://icon4py.github.io",
            "history": output_path.absolute().as_posix()
            + " "
            + dt.datetime.now().isoformat(),  # TODO(halungge): this is actually the path to the binary in ICON not the output path
            "references": "https://icon4py.github.io",
            "uuidOfHGrid": grid_id,
        }
        self.config = config
        self._time_properties = writers.TimeProperties(time_units, calendar)
        self._vertical_size = vertical
        self._horizontal_size = horizontal
        self._field_names = config.variables
        self._handle_output_path(output_path, config.filename)
        # The schedule is always evaluated in steps; a time-delta interval is normalized
        # to steps here, using the model time step.
        self._output_interval_steps = _interval_in_steps(config.output_interval, dtime)
        self._step_counter = 0
        self._file_counter = 0
        self._current_timesteps_in_file = 0
        self._dataset: writers.NETCDFWriter | None = None

    @property
    def output_path(self) -> pathlib.Path:
        return self._output_path

    def _handle_output_path(self, output_path: pathlib.Path, filename: str) -> None:
        file = output_path.joinpath(filename).absolute()
        path = file.parent
        path.mkdir(parents=True, exist_ok=True)
        self._output_path = path
        self._file_name_pattern = file.name

    def _init_dataset(
        self,
        vertical_params: v_grid.VerticalGrid,
        horizontal_size: base.HorizontalGridSize,
    ) -> None:
        """Initialise the dataset with global attributes and dimensions.

        TODO(halungge): as long as we have no terrain it is probably ok to take vct_a as vertical
                          coordinate once there is terrain k-heights become [horizontal, vertical ] field

        """
        if self._dataset is not None:
            self._dataset.close()
        self._file_counter += 1
        filename = generate_name(self._file_name_pattern, self._file_counter)
        filename_path = self._output_path.joinpath(filename)
        # The per-run file counter restarts at 0, so file names (``..._0001.nc``) would
        # collide with -- and silently overwrite -- output from a previous run sharing this
        # directory. Refuse to overwrite: fail loudly so prior results are never lost.
        # TODO (jcanton): take care of this when implementing restart
        if filename_path.exists():
            raise exceptions.InvalidConfigError(
                f"Output file '{filename_path}' already exists; refusing to overwrite output "
                f"from a previous run. Use a fresh output directory."
            )
        df = writers.NETCDFWriter(
            file_name=filename_path,
            vertical=vertical_params,
            horizontal=horizontal_size,
            time_properties=self._time_properties,
            global_attrs=self._global_attrs,
        )
        df.initialize_dataset()
        self._dataset = df

    def store(
        self, state: dict, model_time: dt.datetime, *args: Any, **kwargs: dict[str, Any]
    ) -> None:
        """Pick fields from the state dictionary to be written to disk.

        Args:
            state: dict  model state dictionary
            model_time: the current time step of the simulation
        """
        self._step_counter += 1
        if not self._at_capture_time():
            return
        # TODO(halungge): this should do a deep copy of the data
        try:
            state_to_store = {field: state[field] for field in self._field_names}
        except KeyError as e:
            log.error(f"Field '{e.args[0]}' is missing in state.")
            self.close()
            raise exceptions.IncompleteStateError(e.args[0]) from e

        log.info(f"Storing fields {state_to_store.keys()} at {model_time}")

        if self._do_initialize_new_file():
            self._init_dataset(self._vertical_size, self._horizontal_size)
        self._append_data(state_to_store, model_time)

        self._update_current_file_count()
        if self._is_file_limit_reached():
            self.close()

    def _update_current_file_count(self) -> None:
        self._current_timesteps_in_file = self._current_timesteps_in_file + 1

    def _do_initialize_new_file(self) -> bool:
        return self._current_timesteps_in_file == 0

    def _is_file_limit_reached(self) -> bool:
        return 0 < self.config.timesteps_per_file == self._current_timesteps_in_file

    def _append_data(self, state_to_store: dict, model_time: dt.datetime) -> None:
        assert self._dataset is not None
        self._dataset.append(state_to_store, model_time)

    def _at_capture_time(self) -> bool:
        # fire every N model steps
        return self._step_counter % self._output_interval_steps == 0

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
            self._current_timesteps_in_file = 0


def generate_name(fname: str, counter: int) -> str:
    stem = fname.split(".", maxsplit=1)[0]
    return f"{stem}_{counter:0>4}.nc"

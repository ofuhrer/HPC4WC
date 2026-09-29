# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import datetime as dt
import functools
import logging
import pathlib
import uuid
from typing import Final, Required, TypedDict

import netCDF4 as nc
import numpy as np
import xarray as xr

import icon4py.model.common.states.metadata
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base, vertical as v_grid
from icon4py.model.common.io import cf_utils
from icon4py.model.common.utils import data_allocation as data_alloc


EDGE: Final[str] = "edge"
VERTEX: Final[str] = "vertex"
CELL: Final[str] = "cell"
MODEL_HALF_LEVEL: Final[str] = "half_level"
MODEL_LEVEL: Final[str] = "level"
TIME: Final[str] = "time"

log = logging.getLogger(__name__)
process_properties = decomposition.SingleNodeProcessProperties()


class GlobalFileAttributes(TypedDict, total=False):
    """
    Global file attributes of a ICON generated netCDF file.

    Attribute map what ICON produces, (including the upper, lower case pattern).
    Omissions (possibly incomplete):
    - 'CDI' used for the supported CDI version (http://mpimet.mpg.de/cdi) since we do not support it

    Additions:
    - 'external_variables': variable used by CF conventions if cell_measure variables are used from an external file'
    """

    #: version of the supported CF conventions
    Conventions: Required[str]  # TODO(halungge): check changelog? latest version is 1.11

    #: unique id of the horizontal grid used in the simulation (from grid file)
    uuidOfHGrid: Required[uuid.UUID]

    #: institution name
    institution: Required[str]

    #: title of the file or simulation
    title: Required[str]

    #: source code repository
    source: Required[str]

    #: path of the binary and generation time stamp of the file
    history: Required[str]

    #: references for publication # TODO(halungge): check if this is the right reference
    references: str
    comment: str
    external_variables: str


@dataclasses.dataclass
class TimeProperties:
    units: str
    calendar: str


class NETCDFWriter:
    """
    Writer for netcdf files.

    Writes a netcdf file using netcdf4-python directly. Currently, this seems to be the only way that we can
      - get support for parallel (MPI available) writing
      - the possibility to append time slices to a variable already present in the file. (Xarray.to_netcdf does not support this https://github.com/pydata/xarray/issues/1672)
    """

    def __init__(
        self,
        *,
        file_name: pathlib.Path | str,
        vertical: v_grid.VerticalGrid,
        horizontal: base.HorizontalGridSize,
        time_properties: TimeProperties,
        global_attrs: GlobalFileAttributes,
        process_properties: decomposition.ProcessProperties = process_properties,
    ):
        self._file_name = str(file_name)
        self._process_properties = process_properties
        self._time_properties = time_properties
        self._vertical_params = vertical
        self._horizontal_size = horizontal
        self.attrs = global_attrs
        self.dataset = None

    def __getitem__(self, item: str) -> str:
        assert self.dataset is not None
        return self.dataset.getncattr(item)

    @functools.cached_property
    def num_levels(self) -> int:
        return self._vertical_params.interface_physical_height.ndarray.shape[0] - 1

    @functools.cached_property
    def num_interfaces(self) -> int:
        return self._vertical_params.interface_physical_height.ndarray.shape[0]

    def initialize_dataset(self) -> None:
        self.dataset = nc.Dataset(  # type: ignore [assignment] # dataset is reassigned here
            self._file_name,
            "w",
            format="NETCDF4",
            persist=True,
            parallel=self._process_properties.comm_size > 1,
            comm=self._process_properties.comm,
        )
        assert self.dataset is not None
        log.info(f"Creating file {self._file_name} at {self.dataset.filepath()}")
        self.dataset.setncatts({k: str(v) for (k, v) in self.attrs.items()})
        ## create dimensions all except time are fixed
        self.dataset.createDimension(TIME, None)
        self.dataset.createDimension(MODEL_LEVEL, self.num_levels)
        self.dataset.createDimension(MODEL_HALF_LEVEL, self.num_interfaces)
        self.dataset.createDimension(CELL, self._horizontal_size.num_cells)
        self.dataset.createDimension(VERTEX, self._horizontal_size.num_vertices)
        self.dataset.createDimension(EDGE, self._horizontal_size.num_edges)
        log.debug(f"Creating dimensions {self.dataset.dimensions} in {self._file_name}")
        # create time variables
        times = self.dataset.createVariable(TIME, "f8", (TIME,))
        times.units = self._time_properties.units
        times.axis = cf_utils.COARDS_TIME_COORDINATE_NAME
        times.calendar = self._time_properties.calendar
        times.standard_name = TIME
        times.long_name = TIME
        # create vertical coordinates:
        levels = self.dataset.createVariable(MODEL_LEVEL, np.int32, (MODEL_LEVEL,))
        levels.units = "1"
        levels.positive = "down"
        levels.long_name = "model full level index"
        levels.standard_name = cf_utils.LEVEL_STANDARD_NAME
        levels[:] = np.arange(self.num_levels, dtype=np.int32)

        half_levels = self.dataset.createVariable(MODEL_HALF_LEVEL, np.int32, (MODEL_HALF_LEVEL,))
        half_levels.units = "1"
        half_levels.positive = "down"
        half_levels.long_name = "model half level index"
        half_levels.standard_name = (
            icon4py.model.common.states.metadata.INTERFACE_LEVEL_STANDARD_NAME
        )
        half_levels[:] = np.arange(self.num_levels + 1, dtype=np.int32)

        heights = self.dataset.createVariable("height", np.float64, (MODEL_HALF_LEVEL,))
        heights.units = "m"
        heights.positive = "up"
        heights.axis = cf_utils.COARDS_VERTICAL_COORDINATE_NAME
        heights.long_name = "height value of half levels without topography"
        heights.standard_name = (
            icon4py.model.common.states.metadata.INTERFACE_LEVEL_HEIGHT_STANDARD_NAME
        )
        heights[:] = data_alloc.as_numpy(self._vertical_params.interface_physical_height)

    def append(self, state_to_append: dict[str, xr.DataArray], model_time: dt.datetime) -> None:
        """
        Append the fields to the dataset.

        Appends a time slice of the fields in the state_to_append dictionary to the dataset for the `model_time` expanding the time coordinate by the `model_time`.
        Args:
            state_to_append: fields to append
            model_time: time of the model state

        Returns:

        """
        assert self.dataset is not None
        time = self.dataset[TIME]
        time_pos = len(time)
        time[time_pos] = cf_utils.date2num(model_time, units=time.units, calendar=time.calendar)
        for var_name, new_slice in state_to_append.items():
            standard_name = new_slice.standard_name
            canonical_new_slice = cf_utils.to_canonical_dim_order(new_slice)
            assert standard_name is not None, f"No short_name provided for {standard_name}."
            ds_var = filter_by_standard_name(self.dataset.variables, standard_name)
            if not ds_var:
                dimensions = ("time", *canonical_new_slice.dims)
                new_var = self.dataset.createVariable(
                    var_name, canonical_new_slice.dtype, dimensions
                )
                new_var[0, :] = data_alloc.as_numpy(canonical_new_slice.data)
                new_var.units = canonical_new_slice.units
                new_var.standard_name = canonical_new_slice.standard_name
                new_var.long_name = canonical_new_slice.long_name
                new_var.coordinates = canonical_new_slice.coordinates
                new_var.mesh = canonical_new_slice.mesh
                new_var.location = canonical_new_slice.location

            else:
                assert ds_var is not None
                actual_var_name = ds_var.get(var_name).name
                dims = ds_var.get(actual_var_name).dimensions
                shape = ds_var.get(actual_var_name).shape
                assert len(canonical_new_slice.dims) == len(dims) - 1, (
                    f"Data variable dimensions do not match for {standard_name}."
                )

                # TODO(halungge): change for parallel/distributed case: where we write at `global_index` field on the node for the horizontal dim.
                # we can acutally assume fixed index ordering here, input arrays are  re-shaped to canonical order (see above)

                right = (slice(None),) * (len(dims) - 1)
                expand_slice = (
                    slice(shape[cf_utils.COARDS_T_POS] - 1, shape[cf_utils.COARDS_T_POS]),
                )
                slices = expand_slice + right
                self.dataset.variables[actual_var_name][slices] = data_alloc.as_numpy(
                    canonical_new_slice.data
                )

    def close(self) -> None:
        assert self.dataset is not None
        if self.dataset.isopen():
            self.dataset.close()

    @property
    def dims(self) -> dict:
        assert self.dataset is not None
        return self.dataset.dimensions

    @property
    def variables(self) -> dict:
        assert self.dataset is not None
        return self.dataset.variables


def filter_by_standard_name(model_state: dict, value: str) -> dict:
    return {k: v for k, v in model_state.items() if value == v.standard_name}

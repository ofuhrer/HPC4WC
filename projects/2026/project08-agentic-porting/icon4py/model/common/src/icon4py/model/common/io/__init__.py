# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Module containing the IO functionality for ICON4Py.

### General concept

The module provides an `IOMonitor` that captures fields from the model state and writes them to a file
if called at the configured output time of the field. Upon each call the monitor decides on its own
what fields it needs to write.

The Datafiles produced are NETCDF4 files and conform to
[CF conventions](https://cfconventions.org/cf-conventions/cf-conventions.html).
In addition, upon start-up the monitor writes a copy of the original ICON grid file enhanced with a
[UGRID](https://ugrid-conventions.github.io/ugrid-conventions/) conforming mesh, which is referenced
in the datafiles. This grid file has the same name as the original grid file with the suffix `_ugrid.nc`.

The model state is a dictionary of `xarray.DataArrays` containing the ICON4Py fields as data buffers and
CF conventional metadata. For some basic examples see [data.py](../states/data.py).

#### Adding fields

When adding new fields to the state the `short_name` should be taken from the
[CF standard name table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
or, if not available there, built up according to [guidelines of CF standard names](http://cfconventions.org/Data/cf-standard-names/docs/guidelines.html).

### Usage

#### Configuration

The IO module is configurable and can be configured with:

- `output_path`: path where all output files will be stored.
- `field_groups`: list of field group configuration (see below).
- `time_units` (optional, default is "seconds since 1970-01-01 00:00:00"): unit used with the time dimension in the data files.
- `calendar` (optional, default is "proleptic_gregorian"). Caleandar used with the time dimension in the data files.

Field groups are stored in the same file and share a common setting of

- `output_interval`: the output schedule, given as either a positive integer N (write every N model steps, i.e. every N calls to `store`) or a `datetime.timedelta` (a simulation-time delta, e.g. `timedelta(hours=2)`). A time delta is normalized to a number of steps using the model time step, so the schedule is always evaluated in steps. Defaults to every step.
- `filename`: File name to be used for the datafile, it may contain a _relative_ path which is appended to the `output_path` . Files will be appended with a counter for roll over (see `timesteps_per_file`).
- `timesteps_per_file` (default=10): Number of timesteps to be recorded in one file, if the value is negative all captured times go into the same file.
- `variables`: List of variables names to be output. Variable names are the CF names used as keys in the model state (see [data.py](../states/data.py)).
- `nc_title` (optional): Title field of the generated netcdf file.
- `nc_comment` (optional): Comment to be put to generated netcdf file.

As we have no general handling of configuration files in `ICON4Py` yet, the configuration needs to
be instantiated as Python dataclasses for now. A valid configuration could look like this:

```python
import datetime

prognostic_group = FieldGroupIOConfig(
    output_interval=datetime.timedelta(hours=2),
    filename="icon4py_prognostics",
    timesteps_per_file=12,
    variables=["air_density", "exner_function", "upward_air_velocity"],
    nc_title="prognostics from my experiment",
    nc_comment="Writing prognostic fields data from icon4py ",
)

wind_group = FieldGroupIOConfig(
    output_interval=1,
    filename="icon4py_diagnostics",
    timesteps_per_file=24,
    variables=["eastward_wind", "northward_wind"],
    nc_comment="Writing additional wind fields data from icon4py",
)

io_config = IOConfig(
    output_path="simulation1/output",
    field_groups=[prognostic_group, wind_group],
)
```

This configuration must then be passed to an instance of the `IOMonitor`, which will decide upon a
call to `IOMonitor.store` what fields of the model state need to be written at that time:

```
io_monitor = IOMonitor(io_config)
(...)
monitor.store(model_state, time)
```

### Restrictions

- We only support NETCDF4 files.
- No transformation are applied to any output data: Fields are written with the same unstructured grid resolutions as they are computed.
- Horizontal coordinates the latitude and longitude in radians as provided by the ICON grid file.
- Vertical coordinates are the model levels, there is no transformation to pressure levels.
- Parallel writing is not yet implemented.
- Global attributes of the datafiles and field metadata is only scarcely available and needs to be augmented.

"""

import importlib.util


if not importlib.util.find_spec("xarray"):
    raise RuntimeError("Optional icon4py-common[io] dependencies are missing!")

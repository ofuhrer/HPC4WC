# Standalone driver for ICON4Py

`main.py` contains a simple Python program to run the experimental ICON4Py port.

The driver reads its configuration from a configuration directory, initializes
the grid and model state, and runs the time integration. Which granules are
active (diffusion, solve_nonhydro, tracer advection, microphysics) is determined
by the provided configuration rather than being hardcoded.

It supports both single-node and distributed (MPI) runs. IO output is currently
single-node only and is therefore disabled for MPI runs.

## Installation

See the general instructions in the [README.md](../../README.md) in the base
folder of the repository.

## Usage

```bash
# set environment variables (optional but convenient)
export ICON4PY_ROOT=<path to the icon4py clone>
export GRID_FOLDER=<path to the folder holding grids>
export CONFIG_FOLDER=<path to the configuration directory>

# command line arguments
icon4py-standalone-driver \
    --grid-file-path $GRID_FOLDER/icon_grid_0013_R02B04_R.nc \
    --config-file-path $CONFIG_FOLDER \
    --icon4py-backend gtfn_cpu \
    --output-path $ICON4PY_ROOT/output_path \
    --enable-output
```

## Configuration directory

The driver expects a configuration directory containing the following JSON files:

- `NAMELIST_ICON_output_atm.json`
- `icon_master.namelist.json`
- `NAMELIST_expname.json`

These are generated from the corresponding Fortran namelists and describe the
experiment, the atmosphere setup, and the input parameters.

To generate these from an experiment run with Fortran ICON you can use the [f90nml](https://f90nml.readthedocs.io/en/latest/index.html) Python package to generate the JSON files from the original `NAMELIST` files of the Fortran ICON simulation.
Once the Fortran ICON simulation has finished, there is a folder generated in `<ICON_ROOT>/<BUILD_TYPE>/experiments/<EXPERIMENT_NAME>` that includes the necessary `NAMELIST` files to configure the `ICON4Py` `standalone_driver`.
Using the following instruction you can export the necessary files to their JSON equivalent format:

```
mkdir CONFIG_DIR
f90nml -g config_nml <ICON_ROOT>/<BUILD_TYPE>/experiments/<EXPERIMENT_NAME>/NAMELIST_ICON_output_atm CONFIG_DIR/NAMELIST_ICON_output_atm.json
f90nml -g config_nml <ICON_ROOT>/<BUILD_TYPE>/experiments/<EXPERIMENT_NAME>/icon_master.namelist CONFIG_DIR/icon_master.namelist.json
f90nml -g config_nml <ICON_ROOT>/<BUILD_TYPE>/experiments/<EXPERIMENT_NAME>/NAMELIST_<EXPERIMENT_NAME> CONFIG_DIR/NAMELIST_expname.json
```

Once the above is done you can provide the `CONFIG_DIR` to the `--config-file-path` of the `icon4py-standalone-driver` to configure the simulation the same way as the ICON Fortran one.

Of course you can write the necessary configuration files manually or start by some template files and edit them yourself.

### Required options

- `--grid-file-path`: path to the ICON grid file.
- `--config-file-path`: path to the directory containing the configuration JSON
  files.
- `--icon4py-backend`: GT4Py backend for running the driver. Run with
  `--help` to see the available backends.

### Optional options

- `--output-path`: override the output path from the configuration file.
- `--log-level`: logging level. Possible values are `notset` (default), `debug`,
  `info`, `warning`, `error`, `critical`.
- `--print-distributed-debug-msg`: print debug logging messages from all MPI
  ranks (only effective when `--log-level debug` is set).
- `--enable-output/--no-enable-output`: write prognostic and diagnostic fields
  to output. Defaults to `--no-enable-output`. Disabled automatically in MPI
  runs.

# model driver for Python ICON port

`icon4py_driver.py` contains a simple python program to run the experimental ICON python port. So far the code mostly draws on serialized ICON data until we increasingly can initialize and run the model independently.

Currently, users need serialized data of the pre-computed metric and interpolation coefficients and grid to run the driver.

Currently, it does only diffusion and solve_nonhydro (dry atmosphere with no physics). The configuration for the granules and driver is hardcoded in [icon4py_configuration.py](src/icon4py/model/driver/icon4py_configuration.py). Time step, total integration time, number of substeps, and etc. can be configured there.

The code is meant to be changed and enlarged as we port new parts of the model.

It runs single node or parallel versions. For parallel runs the domain has to be decomposed previously through a full ICON run that generates the necessary serialized data. Test data for runs with 1, 2, 4 nodes are available.

## Installation

See the general instructions in the [README.md](../../README.md) in the base folder of the repository.

## Usage

```bash
export ICON4PY_ROOT=<path to the icon4py clone>
icon4py_driver $ICON4PY_ROOT/testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data --run_path=$ICON4PY_ROOT/output
```

The driver code runs in parallel, in order to do this you need to install the optional communication libraries with:

```bash
cd ICON4PY_ROOT
uv sync --extra distributed  # or `uv sync --extra all` which includes everything

```

then run

```bash
mpirun -np 2 icon4py_driver $ICON4PY_ROOT/testdata/ser_icondata/mpitask2/mch_ch_r04b09_dsl/ser_data --mpi=True --run_path=$ICON4PY_ROOT/output --grid_root=4 --grid_level=9 --experiment_type=any
```

#### Remarks

- First (required) arg is the folder where the serialized input data is stored. In the example above, the path is where the data is put when downloaded via the unit tests (the url can be found in [datatest_utils.py](../common/src/icon4py/model/common/test_utils/datatest_utils.py). You can also generate your own serialized data and put it in an arbitrary folder.
- Parallel runs are possible if corresponding data is provided, which is currently available for test with 2 or 4 MPI processes: check [datatest_fixtures.py](../common/src/icon4py/model/common/test_utils/datatest_fixtures.py) for download urls. However, parallel runs are not yet fully tested.
- Please use the command `icon4py_driver --help` for information on the remaining optional arguments,.

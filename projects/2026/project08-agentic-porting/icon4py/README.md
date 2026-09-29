[![Open in Gitpod](https://img.shields.io/badge/Gitpod-ready--to--code-908a85?logo=gitpod)](https://gitpod.io/#https://github.com/C2SM/icon4py)
[![Nox](https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg)](https://github.com/wntrblm/nox)
[![Open the docs](https://readthedocs.org/projects/docs/badge/?version=latest)](https://c2sm.github.io/icon4py)

# ICON4Py

This repository hosts a work-in-progress Python implementation of the ICON climate and weather model. Additionally, it includes `icon4py.tools`, a collection of command-line interfaces (CLIs), and utilities required for the integration of ICON4Py code into the ICON Fortran model. ICON4Py leverages [GT4Py](https://github.com/GridTools/gt4py) to ensure efficient and performance portable implementations of these components.

## Project Structure

The repository is organized as a _monorepo_, where various ICON model components and utilities are developed as independent Python namespace packages in subfolders. The `icon4py` root package collects specific versions of the different components as dependencies. `icon4py` is published on PyPI as a meta-package. Individual namespace packages are also available on PyPI and can be installed independently.

## License

ICON4Py is licensed under the terms of the BSD-3-Clause.

## Installation Instructions

All ICON4Py packages can be installed through PyPI with the [`icon4py` meta package](https://pypi.org/project/icon4py/).

We recommend using [`uv`](https://docs.astral.sh/uv/) for both regular installation and development workflows. See the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/) for details on installing `uv`.

```bash
uv pip install --prerelease=allow icon4py
```

**Note:** `--prerelease=allow` is currently required for the `dace` dependency, which is pinned to a prerelease version.

### Extras

The `icon4py` meta-package provides the following optional extras. The same extras apply in development workflows by using `uv sync --extra <name1> --extra <name2> ...`.

| Extra         | Description                                                                                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `all`         | All extras listed below, except the GPU extras.                                                                                                        |
| `cuda12`      | Required for CUDA support with CUDA version 12. Requires the CUDA toolkit to be installed.                                                             |
| `cuda13`      | Same as `cuda12` for CUDA version 13.                                                                                                                  |
| `rocm7`       | Same as `cuda*` for ROCm on AMD GPUs.                                                                                                                  |
| `distributed` | Enables support for distributed multi-node runs. Requires an MPI implementation and Boost headers. See below for fixes to installation issues on Alps. |
| `fortran`     | Enables `icon4py-bindings` as a dependency, which provides an interface between ICON4Py and ICON Fortran. Requires C and Fortran compilers.            |
| `io`          | Enables I/O dependencies such as NetCDF.                                                                                                               |
| `testing`     | Enables `icon4py-testing` as a dependency, providing utilities for testing ICON4Py.                                                                    |
| `profiling`   | Enables `viztracer` as a dependency for profiling.                                                                                                     |

For example, to install icon4py with MPI and CUDA support:

```bash
uv pip install "icon4py[distributed,cuda12]"
```

### ICON4Py Development Environment

Once `uv` is installed in your system, it is enough to clone this repository and let `uv` handling the installation of the development environment.

**Important**: the `uv sync` command should always be executed from the **root folder** of the repository, to make sure it installs all the workspace dependencies and not only the dependencies of a subproject.

The `--extra distributed` option installs mpi4py and ghex, which require Boost headers and an MPI implementation (e.g. OpenMPI) to be available on the system.

```bash
# Clone the repository
git clone git@github.com:C2SM/icon4py.git
cd icon4py

# Let uv create the development environment at `.venv`.
# The `--extra all` option tells uv to install all the optional
# dependencies of icon4py, and thus it is not strictly necessary.
# Note that if no dependency groups are provided as an option,
# uv uses `--group dev` by default so the development dependencies
# are also installed. 
uv sync --extra all

# Activate the virtual environment and start writing code!
source .venv/bin/activate
```

The new _venv_ is a standard Python virtual environment preconfigured with all necessary runtime and development dependencies. Additionally, all icon4py subpackages are installed in editable mode, allowing for seamless development and testing.

To install new packages, use the `uv pip` subcommand, which emulates the `pip` interface and is generally much faster. Alternatively, the standard `pip` command is also available within the venv, although using `pip` directly is slower and not recommended.

The `pyproject.toml` file at the root folder contains both the definition of the `icon4py` Python distribution package and the settings of the development tools used in this project, most notably `uv`, `ruff`, `mypy` and `pytest`. It also contains _dependency groups_ (see [PEP 735](https://peps.python.org/pep-0735/) for further reference) with the development requirements listed in different groups (`build`, `docs`, `lint`, `test`, `typing`, ...) and collected together in the general `dev` group which gets installed by default by `uv`.

## Development Instructions

By following the installation instructions above, the source files are imported directly by the Python interpreter meaning that any code change is available and executed by the interpreter.

To add new dependencies to the project, either core/optional run-time or development-only dependencies, it is possible to use the `uv` cli direcly or to modify by hand the appropriate tables in the corresponding `pyproject.toml` (check `uv` documentation for more information [https://docs.astral.sh/uv/concepts/projects/dependencies/](https://docs.astral.sh/uv/concepts/projects/dependencies/)).

### Code Quality Checks

[pre-commit](https://pre-commit.com/) is used to run several linting and checking tools. It should always be executed locally before opening a pull request. When executing pre-commit locally you can either run it for the `model` or `tools` folder:

For example to run code checks on all components in `icon4py.model` you can do:

```bash
# running precommit for all components in model
cd model/
pre-commit run --all-files
```

### Testing

[pytest](https://pytest.org/) is used for testing and it also comes with a command tool to easily run tests:

```bash
# Run all tests (verbose mode)
pytest -v

# Run only tests in `path/to/test/folder`
pytest -v path/to/test/folder
```

`nox` is recommended for running comprehensive test suites across multiple Python versions and configurations, mirroring the setup used in the CI pipeline.

```bash
# List all available test sessions (colored items are the default sessions)
nox -l

# Run all parametrized cases of a session
nox -s 'test_common'

# Run a test session for a specific python version and parameter value
nox -s 'test_atmosphere_advection-3.10(datatest=True)'
```

To run distributed tests, make sure an MPI implementation is installed and run `uv sync --extra distributed` or `uv sync --extra all`. Then run tests using `mpirun`, the `--with-mpi` pytest flag, and the `-k mpi_tests` filter:

```bash
mpirun -np 4 pytest -v -s --with-mpi -k mpi_tests
```

To avoid all ranks writing their test output to stdout, use the helper script `ci/scripts/ci-mpi-wrapper.sh` around the `pytest` command:

```bash
mpirun -np 4 ci/scripts/ci-mpi-wrapper.sh pytest -v -s --with-mpi -k mpi_tests
```

#### Distributed runs on Alps

##### Compiling

On Alps with Cray MPICH and GH200 or A100 GPUs, install mpi4py and GHEX as follows:

```bash
export GHEX_USE_GPU=ON
export GHEX_GPU_TYPE=NVIDIA
export GHEX_GPU_ARCH="80;90"
export GHEX_TRANSPORT_BACKEND=MPI
export MPICH_CXX=$(which g++)
export MPICH_CC=$(which gcc)
uv sync --no-binary-package mpi4py --extra all --extra distributed --extra cuda12 --python $(which python) --refresh
```

If you already have a broken GHEX, mpi4py, or other package in the uv cache, run the command with `--no-cache` after either uninstalling the broken package or wiping the virtualenv.

`--no-binary-package mpi4py` is required because Cray MPICH is not ABI compatible with the MPI used to build mpi4py binary wheels. If you don't do this you may get an error like:

```
ImportError: libmpi.so.12: cannot open shared object file: No such file or directory
```

when importing mpi4py. The `GHEX_*` options tell GHEX to build with GPU support. If you don't, you may see errors like:

```
AttributeError: module 'ghex.pyghex' has no attribute 'unstructured__data_descriptor_gpu_int_int_double_'
```

when GHEX tries to perform halo exchanges. The `MPICH_*` options make sure mpi4py gets built with GCC instead of NVHPC. mpi4py assumes that it can set certain compiler flags that GCC supports, but NVHPC does not support. The error message will typically look like:

```
      [stderr]
      nvc-Error-Unknown switch: -fwrapv
      error: Cannot compile MPI programs. Check your configuration!!!
      Installing mpi4py requires a working MPI implementation.
      If you are running on a supercomputer or cluster, check with
      the system administrator or refer to the system user guide.
      Otherwise, if you are running on a laptop or desktop computer,
      your may be missing the MPICH or Open MPI development package:
```

##### Running

When running make sure to

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
```

Cray MPICH will otherwise segfault when communicating GPU buffers. Also see the [CSCS Cray MPICH documentation](https://docs.cscs.ch/software/communication/cray-mpich/) for more details.

When performance is important you may want to

```bash
export PYTHONOPTIMIZE=2
```

### Benchmarking

We use [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/) to benchmark the execution time of stencils in icon4py. To disable benchmarking during testing you can use `--benchmark-disable` when invoking `pytest`.

### Documentation

Documentation is at a very early stage given the constant state of development.
Ongoing efforts to document the dycore can be viewed at [c2sm.github.io/icon4py](https://c2sm.github.io/icon4py).

You can install the required packages by using the provided `docs` dependency group, which is included in the `dev` installed by default by `uv sync` if no dependency groups are specified.

To build the html documentation in your local repository copy starting from the most upwards directory, apply the following commands:

```bash
cd model/atmosphere/dycore/docs
make html
```

The local documentation could then be accessed at `docs/_build/html/index.html`

### More Information

For more information please consult the package specific READMEs found in the `model` and `tools` folders.

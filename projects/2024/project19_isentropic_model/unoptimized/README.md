# A mountain flow model in Python

This repository contains the full mountain flow model in Python for the course
"Weather and Climate Models" @ ETH Zürich.

## Installation

The model requires at least python version 3.6.7. In the following, we show the steps to install python 3.12.2 (which is currently the latest version, as of April 2024).

### Model

The model comes in the shape of a Python package called `nmwc_model` whose source
files are in the `nmwc_model` directory.

**Note.** It is always advisable to install Python packages in a dedicated virtual
environment, not to pollute any system-wide installation. If you are using
[virtualenv](https://virtualenv.pypa.io/en/latest/), you may want to use the
provided `bootstrap_venv.sh` to automate the creation of a virtual environment.
If you are using [conda](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf):

The package `nmwc_model` and all its dependencies
(listed in `requirements.txt`) can be installed as follows:

Linux (MacOS follows similar procedure):

    # using conda (if you installed Anaconda, or if you installed conda some other way)
    	# navigate to the location of the (extracted) ModelPython folder
    	# for example:
    	cd /home/<user>/Downloads/ModelPython

    	# check location
    	ls (should return a list of files containing the README.md)

    	# create a conda virtual environment
    	conda create --name venv-wcm

    	# activate the virtual environment
    	conda activate venv-wcm

    	# install python (also installs the python package manager pip)
    	conda install python

    	# install the two dimensional isentropic model
    	pip install --upgrade setuptools wheel
    	pip install -e .

    	# check installation
    	python test_installation.py (should return Installation completed successfully!)

    	# if you need to deactivate the environment
    	conda deactivate

    # only using python (without conda)
    	# open a new terminal
    	# navigate to the location of the (extracted) ModelPython folder
    	# for example:
    	cd /home/<user>/Downloads/ModelPython

    	# check the location
    	ls (should return a list of files containing the README.md)

    	# create a virtual environment
    	python -m venv venv_wcm

    	# activate the virtual environment
    	source venv_wcm/bin/activate

    	# install the two dimensional isentropic model
    	pip install --upgrade setuptools wheel
    	pip install -e .

    	# check installation
    	python test_installation.py (should return Installation completed successfully!)

    	# if you need to deactivate the environment
    	deactivate

Windows:

    # using conda (if you installed Anaconda, or if you installed conda some other way)
    	# navigate to the location of the (extracted) ModelPython folder
    	# for example:
    	cd C:\Users\<user>\Downloads\ModelPython

    	# check location
    	ls (should return a list of files containing the README.md)

    	# create a conda virtual environment
    	conda create --name venv-wcm

    	#show my environments:
    	conda env list

    	# activate the virtual environment
    	conda activate venv-wcm

    	# install python (also installs the python package manager pip)
    	conda install python

    	# install the two dimensional isentropic model
    	pip install --upgrade setuptools wheel
    	pip install -e .

    	# check installation
    	python test_installation.py (should return Installation completed successfully!)

    	# if you need to deactivate the environment
    	conda deactivate

    # only using python (without conda)
    	# download [python 3.12.2 64bit](https://www.python.org/ftp/python/3.12.2/python-3.12.2-amd64.exe)
    	# FIRST: look at the bottom and check the box Add python.exe to PATH
    	# THEN: click on Install Now

    	# open the Windows PowerShell (just search for PowerShell and open it)
    	# check that python installed correctly
    	python --version (should return Python 3.12.2)

    	# navigate to the location of the (extracted) ModelPython folder
    	# for example:
    	cd C:\Users\<user>\Downloads\ModelPython

    	# check location
    	ls (should return a list of files containing the README.md)

    	# create a python virtual environment
    	python -m venv venv_wcm

    	# activate the virtual environment
    	venv_wcm\Scripts\Activate.ps1

    	# install the two dimensional isentropic model
    	pip install --upgrade setuptools wheel
    	pip install -e .

    	# check installation
    	python test_installation.py (should return Installation completed successfully!)

    	# if you need to deactivate the environment
    	deactivate

### Getting started with Spyder

[Anaconda](https://www.anaconda.com/distribution/) comes with the Spyder IDE.
Here the steps to perform to make `nmwc_model` work in Spyder.

1. First change the working directory (visible in the upper right corner) to the
   repository root folder.
2. The IPython interactive console (bottom right corner) should get automatically
   redirected to the new working directory. You can check this by typing `cd`.
3. Install the `nmwc_model` package by typing `!conda develop .` (please note that
   the exclamation mark is needed!).
4. Check if the installation was successful by running the script
   `test_installation.py`.

**Remark.** Any Python script can be run from within Spyder either clicking on
the rightward green triangle in the toolbar, pressing F5, or issuing `%run <script.py>`
in the IPython console.

## Running code

Note that paths are specified with forward slashes (/) in Linux and MacOS, but with backward slashes in Windows (\).

Configuration parameters are picked from `nmwc_model/namelist.py`.

Simulations can be launched as follows:

    # navigate to the location of the (extracted) ModelPython folder
    # for example:
    cd ~/Downloads/ModelPython

    # run the time integration
    python nmwc_model/solver.py

    # run the plotting scripts
    python nmwc_model\xzplot.py
    python nmwc_model\hovx_vel.py
    python nmwc_model\hovz_vel.py

## Tests

Unit tests are placed in the `tests/` folder. Each test covers a specific module
of the code base. So `tests/test_<module>.py` targets `nmwc_model/<module>.py`.

To run the entire test suite:

    # navigate into the `tests/` folder
    cd tests

    # to launch all tests at once
    pytest

    # to run the test for `nmwc_model/<module>.py`:
    pytest test_<module>.py

    # to run a specific test function:
    pytest test_<module>.py::<function>

    # exit the `tests/` folder
    cd ..

Each test function is intended to validate the implementation of only one target
function in the source code. Input and validation data are loaded from a `.npz`
dataset stored in `tests/baseline_datasets/`.

The following table lists the source functions which the students are expected
to complete in each tutorial, and the corresponding test functions which
they can use to verify the correctness of their input.

| Tutorial | Target function                          | Test function                                    |
| -------- | ---------------------------------------- | ------------------------------------------------ |
| Ex. 2.1  | `nmwc_model.prognostics.prog_isendens`   | `test_prognostics.py::test_prog_isendens`        |
| Ex. 2.1  | `nmwc_model.prognostics.prog_velocity`   | `test_prognostics.py::test_prog_velocity`        |
| Ex. 2.2  | `nmwc_model.diagnostics.diag_montgomery` | `test_diagnostics.py::test_montgomery`           |
| Ex. 2.2  | `nmwc_model.diagnostics.diag_pressure`   | `test_diagnostics.py::test_pressure`             |
| Ex. 3.3  | `nmwc_model.makesetup.make_profile`      | `test_makesetup.py::test_makeprofile_dry`        |
|          |                                          | `test_makesetup.py::test_makeprofile_dry_shear`  |
| Ex. 4.1  | `nmwc_model.makesetup.make_profile`      | `test_makesetup.py::test_makeprofile_moist`      |
| Ex. 4.1  | `nmwc_model.prognostics.prog_moisture`   | `test_prognostics.py::test_prog_moisture`        |
| Ex. 5.1  | `nmwc_model.prognostics.prog_numdens`    | `test_prognostics.py::test_prog_numdens`         |
| Ex. 5.2  | `nmwc_model.prognostics.prog_isendens`   | `test_prognostics.py::test_prog_isendens_idthdt` |
| Ex. 5.2  | `nmwc_model.prognostics.prog_moisture`   | `test_prognostics.py::test_prog_moisture_idthdt` |
| Ex. 5.2  | `nmwc_model.prognostics.prog_numdens`    | `test_prognostics.py::test_prog_numdens_idthdt`  |
| Ex. 5.2  | `nmwc_model.prognostics.prog_velocity`   | `test_prognostics.py::test_prog_velocity_idthdt` |

# Euler

    sbatch -n 1 --time=00:01:00 --mem-per-cpu=4096 --wrap="python -m cProfile nmwc_model/solver.py" -o unoptimized.txt

# Profiling

    python -m cProfile -s 'cumulative' nmwc_model/solver.py >> profile_macbookair_modified_namelist.txt

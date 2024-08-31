# High-level, Mid-level, and Low-level GPU Programming Comparison

**Authors:** [Marco Julian Solanki](https://github.com/marcosolanki), [Thibault Meier](https://github.com/tibo1291), and [Sebastian Heckers](https://github.com/seba-heck).

_**Task:** Implement a CUDA and OpenACC version of stencil2d. Validate your results. Compare performance against the CuPy and GT4Py versions of the code for a single GPU socket for different domain sizes. If time permits, analyse and optimise the performance of the implementations._

The latest versions of the here-featured codes can always be retrieved from [this project's GitHub page](https://github.com/marcosolanki/high-performance-computing-for-weather-and-climate).


# Building and running instructions

## CUDA
### Locally:
Install [Nvidia's CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and make sure `nvcc` is in your `PATH`.\
Build with: `make`.\
Run, e.g., with: `./main 128 128 64 2 1024 laplap-global`.\
Clean with: `make clean`.

### On Piz Daint:
Make sure `PrgEnv-nvidia` is loaded.\
Build with: `make`.\
Run, e.g., with: `srun -A class03 -C gpu ./main 128 128 64 2 1024 laplap-global`.\
Clean with: `make clean`.

## OpenACC
### Locally:
Install [Nvidia's HPC SDK](https://developer.nvidia.com/hpc-sdk) and make sure `nvc++` is in your `PATH`.\
Build with: `make`.\
Run, e.g., with: `./main 128 128 64 2 1024 parallel`.\
Clean with: `make clean`.

### On Piz Daint:
Make sure `PrgEnv-nvidia` is loaded.\
Build with: `make`.\
Run, e.g., with: `srun -A class03 -C gpu ./main 128 128 64 2 1024 parallel`.\
Clean with: `make clean`.

## CuPy
### Locally:
Create a new virtual environment: `python -m venv .venv`.\
Activate the environment: `source .venv/bin/activate`.\
Install the dependencies: `pip install -r requirements.txt`.\
Run, e.g., with: `python main.py -nx 128 -ny 128 -nz 64 -bdry 2 -itrs 1024`.

### On Piz Daint:
Make sure you have the `HPC4WC_kernel` IPython kernel from the [HPC4WC setup script](https://github.com/ofuhrer/HPC4WC/blob/main/setup/HPC4WC_setup.sh) installed.\
Open `run.ipynb` via the [CSCS JupyterHub](https://jupyter.cscs.ch).\
Run a cell such as: `!python main.py -nx 128 -ny 128 -nz 64 -bdry 2 -itrs 1024`.

## GT4Py
### Locally:
_**Note:** GT4Py is currently incompatible with the latest version of Python (v3.12)._\
_Therefore, `python3.10` is used in the following._

Create a new virtual environment: `python3.10 -m venv .venv`.\
Activate the environment: `source .venv/bin/activate`.\
Install the dependencies: `pip install -r requirements.txt`.\
Run, e.g., with: `python xyz-laplap.py -nx 128 -ny 128 -nz 64 -bdry 2 -itrs 1024 -bknd cuda`.

### On Piz Daint:
Make sure you have the `HPC4WC_kernel` IPython kernel from the [HPC4WC setup script](https://github.com/ofuhrer/HPC4WC/blob/main/setup/HPC4WC_setup.sh) installed.\
Open `run.ipynb` via the [CSCS JupyterHub](https://jupyter.cscs.ch).\
Run a cell such as: `!python xyz-laplap.py -nx 128 -ny 128 -nz 64 -bdry 2 -itrs 1024 -bknd cuda`.


# Testing and verification

## plot.py
The script `scripts/plot.py` plots the input and output fields using the `in_field.npy`/`in_field.csv` and `out_field.npy`/`out_field.csv` files located within the directory from which it is called. These `*.npy`/`*.csv` files are automatically generated when running any of the provided code versions (`cuda`/`openacc`/`cupy`/`gt4py`). A possible workflow would, therefore, be:
```
cd cuda/ && make && ./main 128 128 64 2 1024 laplap-global
python ../scripts/plot.py
```

## verify.py
The script `scripts/verify.py` compares the `in_field.npy`/`in_field.csv` and `out_field.npy`/`out_field.csv` files produced by two different code versions. It computes the error between them, plots this error, and computes its $L^1$, $L^2$ and $L^\infty$ norms. A possible workflow would, therefore, be:
```
cd cuda/ && make && ./main 128 128 64 2 1024 laplap-global && cd ../
cd openacc/ && make && ./main 128 128 64 2 1024 parallel && cd ../
python scripts/verify.py cuda openacc
```

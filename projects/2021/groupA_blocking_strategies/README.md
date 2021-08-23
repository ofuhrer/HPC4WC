# HPC4WC-Project
![Build](https://github.com/maede97/HPC4WC-Project/actions/workflows/build.yml/badge.svg)
![UnitTests](https://github.com/maede97/HPC4WC-Project/actions/workflows/unittests.yml/badge.svg)
![Documentation](https://github.com/maede97/HPC4WC-Project/actions/workflows/documentation.yml/badge.svg)

## Attention!
This is only a copy of the real repository. The automatic unittests and documentation will not work!

Please visit [https://github.com/maede97/HPC4WC-Project](https://github.com/maede97/HPC4WC-Project) for the actual git repository of the project.

## Requirements
- CMake >3.11
- C++ 17/20
- Python 3

## Build Instructions
Run the following commands, after you have cloned the repository:
```
mkdir build
cd build
cmake ..
make
```
The CMake infrastructure will automatically download all necessary libraries.

## Documentation
The documentation is built using doxygen and publicly available: [https://maede97.github.io/HPC4WC-Project](https://maede97.github.io/HPC4WC-Project)

## Unittests
Unittests are automatically built and run on Windows and Ubuntu, both in Release and Debug mode. See [folder tests](https://github.com/maede97/HPC4WC-Project/tree/master/tests).

After you have completed the build instructions, run the unittest binary using
```
./tests/unittests
```
from within the `build` folder.

## Piz Daint
Our code is designed that it can be run on [Piz Daint](https://www.cscs.ch/computers/piz-daint/). To enable everything, log in and clone this repository. Then, run the following commands:

```
cd scripts
bash create_env.sh
bash cmake.sh
```

This will create a virtual environment using python and installing a suitable CMake version. Afterwards, `cmake.sh` will load the environment and start the cmake-process. Running `make` can then be done without the virtual environment being active. If you have to active it yourself, execute the following two lines:
```
cd  ~
source hpc_env/bin/activate
```

## Reproducing Results
In order to reproduce the results on Piz Daint, create the environment as mentioned above and execute the following lines:
```
cd scripts/slurm
bash block_ij.sh
```
Remember to change the parameters inside the `src/executables/block_ij.cpp` binary.

In order to run the compile-time versions, checkout the `compile-time` branch and there execute `block_ij_compiletime.py` from the `scripts` folder (and change the parameters in it and the parameters inside of `src/executables/diffusion2d.cpp`).

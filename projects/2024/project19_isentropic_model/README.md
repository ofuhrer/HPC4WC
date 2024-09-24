# Parallelization of an Isentropic model with mpi4py

This repository contains all relevant files for the parallelization project conducted during the spring term of 2024 for the «High Performance Computing for Weather and Climate» course at ETH Zurich. In particular, you will find the following relevant files:

- The final report can be read by opening [report-isentropic-model.pdf](report-isentropic-model.pdf).

- The folder [unoptimized/](unoptimized/) contains the original isentropic model, originally developed by Christoph Schär.

- Inside [optimized/](optimized/) you will find our adaptation of the model that supports parallelization using the MPI standard. This was achieved by overhauling the entrypoint of the simulation, [nmwc_model_optimized/solver.py](optimized/nmwc_model_optimized/solver.py), and adapting its dependencies. Notably, [nmwc_model_optimized/parallel.py](optimized/nmwc_model_optimized/parallel.py) was created and contains some helper functions that were developed to successfully perform parallelization.

- The root [tests/](tests/) directory contains tests to test the aforementioned parallelization helper functions. (The `tests/` directories inside the `optimized` and `unoptimized` folders are part of the pre-existing model.)

- The raw data for all benchmarking results that were used in the final report can be found inside [measurements/](measurements/).

Furthermore, you will find the following Jupyter notebooks:

- [profiling.ipynb](profiling.ipynb) was used to profile the unoptimized model.

- [run-models.ipynb](run-models.ipynb) allows you to run both the optimized and unoptimized models. Warning: If you want to execute the optimized model on your local machine, you might encounter an error;  please use the `mpirun` CLI instead. The printed computation times were used to create the files inside [measurements/](measurements/).

- [benchmarking.ipynb](benchmarking.ipynb) performs the actual benchmarking by taking the data in [measurements/](measurements/) and evaluating it.

### Testing the parallelization helper functions

In order to ensure that the methods used during the parallelization make sense, we test them by constructing an integer array on each rank, applying the parallelization helper function and then comparing the resulting arrays to the expected outcome. In order to test, we install the required dependencies:

    pip install -r optimized/requirements.txt
    pip install -e optimized
    pip install pytest numpy 

And run the tests on each rank using mpirun and pytest:

    mpirun -n 4 pytest -q tests/test_exchange_borders.py
    mpirun -n 4 pytest -q tests/test_gather_1d.py
    mpirun -n 4 pytest -q tests/test_gather_2d.py
# Correctness
To test whether your implementation is correct (with respect to the baseline of the course), you first need to save the result of your version into a file. You can use and probably should use `np.save()` such that the results are consistent and for that matter it probably makes sense to convert the output of your implementation in a specific data model to a `numpy` array. You should name the file as follows `f"{path}/{datetime.now().strftime("%Y%m%dT%H%M%S")}-nx{nx}_ny{ny}_nz{nz}_iter{num_iter}_halo{num_halo}_p{precision}.npy"` such that the solution checker knows which solution to compare your output to and generate the necessary one if it does not exist yet. Once you have the file with your output, you can run the following script as follows to check whether it gives the correct result:
```bash
python3 scripts/check_solution.py -s <path-to-data>
```

You should always specify the paths relative to the location of the script and not from where you are runnign the script.

# Measurements
To take measurements the script `scripts/tester.py` can be used. The different domains, functions, number of iterations and precision can be specified. The script saves intermediary results to the folder `results_tmp ` from which the last results must be manually copied to some other location as `results_tmp` is overridden at the start of `scripts/tester.py`

# Implementations
There are various different implementations:
- `baseline`: Slightly adapted version from course to serve as a baseline
- `jax_base`: A basic adaption of `baseline` that uses jax
- `jax`: Use the JAX JIT compiler
- `numba_improved`: An improved Numba version
- `numba_stencil_vectorize`: A Numba version that uses the stencil and vecorize decorator
- `numba`: Adaption of `baseline` to use Numba
- `numpy`: Add some input arguments, else the same as `baseline`
- `torch_try_conv`: Torch version using convolutional neural network
- `torch`: Adaption of `baseline` to use Torch

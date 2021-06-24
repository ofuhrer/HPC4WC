# src

## run

Each function can be called through its main function via: <br/>

`heat_3d_name.py --nx int --ny int --nz int --nt int --plot_result bool`<br/>

`nx, ny, nz` are the number of grid points in each dimension, nt the number of iterations and --plot_result decides whether a plot is stored<br/>

## requirements

### heat3d_np

Uses `numpy`.

### heat3d_cp

Uses `numpy` as well as `cupy`, an open-source array library which makes use of NVIDIA CUDA.

### heat3d_gt4py

Uses `numpy` and `gt4py`, a set of libraries to write and compile hardware independent code.

### heat3d_devito

Uses `numpy` and `devito`, a domain specific language with the aim to write optimized finite difference schemes. Based on `sympy`, a library for symbolic computing in `python`.

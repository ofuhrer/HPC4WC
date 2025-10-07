# 2D Stencil (4th‑order diffusion) — NumPy / Numba / Torch / JAX (+bench & plots)

This project implements the same numerical kernel in four libraries and compares them:

- NumPy (vectorized slicing; to be found in: `stencil2d_new.py`)
- Numba (nopython JIT loops; to be found in: `stencil2d_numba_new.py`)
- Torch (tensor ops with `no_grad`, we only looked at CPU performance; to be found in: `stencil2d_torch_new.py`)
- JAX (functional + `jax.jit`, we only looked at CPU performance; to be found in: `stencil2d_jax_new.py`)

We also include a benchmark harness (`stencil_bench.py`) that runs all drivers across grid sizes and repetitions, plus a post‑processing script (`overlay_and_speedup.py`) that overlays medians and computes speedups.

Here is a Run cheatsheet, to get all the scripts running. Every script has default versions, so we can run all scripts just by using:
> python script_name.py

The scripts can also handle run commands via terminal, some example for each script are provided here:
> python stencil2d_new.py --nx 128 --ny 128 --nz 64 --num_iter 128 --num_halo 2 --plot_result
> python stencil2d_numba_new.py --nx 128 --ny 128 --nz 64 --num_iter 128 --num_halo 2 --plot_result
> python stencil2d_torch_new.py --nx 128 --ny 128 --nz 64 --num_iter 128 --num_halo 2 --device cpu --plot_result
> python stencil2d_jax_new.py --nx 128 --ny 128 --nz 64 --num_iter 128 --num_halo 2 --device cpu --plot_result
> python stencil_bench.py --programs numpy numba torch jax --sizes 32 48 64 96 128 192 --iters 128 --reps 50
> python overlay_and_speedup.py

For torch and jax, there is the possibility to run it on GPU by switching from '--device cpu' to '--device cuda', but then we first need to install CUDA, since it wasn't already compatible to run. Which we didn't do in this project. We focused on CPU.

# Stack

We encountered some problems when running Numba several times over the stencil_bench.py srcipt. According to ChatGPT we had a problem, because the Numab version we used wasn't ABI compatible with the Numpy version. With ChatGPTs advice we pinned the Numpy with `pip install --upgrade "numpy==2.1.3"` to a compatible version in our environment and unparalleld the `stencil2d_numba_new.py` script to avoid problematic parallel backend to crash and it is able to run with these configurations. The disadvantage is, Numba could be a bit slower than its potential, since it isn't in parallel anymore.


# Behind the scences - what is achieved by each stencil2d script
First some common flags for each script:
- nx, --ny, --nz: grid sizes without halos (the code adds 2*h in x and y for halos).
- num_iter: how many diffusion steps to run.
- num_halo: halo width (default 2, required here because the 4th-order scheme needs enough halo cells).
- plot_result: saves a picture (middle z-slice) of the input and/or output field.

All four drivers apply a 4th‑order diffusion to a 3D field field[z, y, x] using periodic halos of width h (default 2). Per iteration:

1. Periodic halos are updated (bottom/top, then left/right; corners via the second phase).
2. `tmp  = Lap(in, extend=1)` — 2D (x,y) 5‑point Laplacian one cell into the halo.
3. `lap2 = Lap(tmp, extend=0)` — Laplacian on the interior only.
4. AXPY update on the interior: `out = in - alpha * lap2` with `alpha = 1/32` (float32).
5. Swap role of input/output buffers between iterations (pointer swap). After the loop, update the halo of the final buffer.

All drivers print a single parseable line:
```
Elapsed time for work = 0.123456 s
```
and Outputs:
- Arrays: in_field_numpy.npy, out_field_numpy.npy
- PNGs (middle z-slice): in_field_numpy.png, out_field_numpy.png

# Drivers in detail

# 1) NumPy — `stencil2d_new.py`
is a pure-NumPy version that uses array slicing to express the stencil. Everything stays in float32 instead of default float64 from original `stecil2d.py` script we got in the course to match fortrans wp=4, which now leads to less memory traffic. A recommondation we took from ChatGPT.

Concept:
- We keep halo cells around the domain to implement periodic boundaries by copying edges.
Each iteration does:
- update_halo –> copy edges into halos (wrap-around).
- laplacian(in, tmp, extend=1) –> compute a 5-point Laplacian one cell into the halo.
- laplacian(tmp, out, extend=0) –> compute the Laplacian only in the interior.
- out = in − α * lap2 in the interior (α = 1/32).
- Swap buffers for the next iteration. Final halo update at the end.
- The right-hand side of the Laplacian assignment creates a temporary sum; we write the result into a preallocated output slice to reduce allocations.
- Pointer swapping means odd/even iteration counts change where the result lands; the script ensures out_field holds the final answer.

# 2) Numba — `stencil2d_numba_new.py`

Same algorithm as before, but the core work is in tight Python loops compiled with @njit -> @njit loop kernels for Laplacian, halo updates, and AXPY; warmup call compiles; timed call runs compiled code.

Concept:
- same as with NumPy, but with explicit loops
- first warmup call triggers JIT compilation; timing happens after that.
- As mentioned in # Stack, we used @njit(parallel=False), so these loops are single-threaded Numba kernels.
- CLI: same flags as NumPy.
- I/O: PNGs only when `--plot_result` (`in_field_numba.png`, `out_field_numba.png`).

# 3) Torch — `stencil2d_torch_new.py`

We did the same steps as described above, this time implemented with Torch tensor slicing (very similar to NumPy) and in-place writes and with @torch.no_grad() so autograd overhead is off.

To run it on GPU, it requires working CUDA build on PyTorch, which as mentioned before we didn't prepare to do.


# 4) JAX — `stencil2d_jax_new.py`

Here we used a functional implementation with jax.jit, which again can target CPU or GPU. But rather than modifying in place, Jax returns a new array. But other than that the algorithm is identical.

Concept:
- JAX arrays are immutable -> we build results with .at[...].set(...) to get a new array.
- The main loop is JIT-compiled (using lax.fori_loop), and the first warmup call triggers compilation.
- For accurate timings, the code calls .block_until_ready() to synchronize before stopping the timer.

# Benchmark script to run all scripts — `stencil_bench.py`

With this script we wanted to simplify the analysis. Here we run our drivers (NumPy / Numba / Torch / JAX) for a set of grid sizes many times, parse the printed runtimes, and produce:
- a JSON file with every measurement,
- a boxplot per framework (runtime per gridpoint vs working-set size),
- a summary file with the arguments we used.

Concept for each selected framework and each size n (=nx=ny):
- Builds command to run driver (NumPy, Numba, Torch, Jax)
- runs command reps times (for boxplots in the report we used 100 reps) and lookes for: `Elapsed time for work = <seconds> s` in drivers print
- converts that number into runtime per gridpoint (`rpg_us = (elapsed_seconds / (nx * ny * nz)) * 1e6`)
- then it stores every run and creates a boxplot over the repetitions for each size

Key options
- `--programs numpy numba torch jax` select frameworks to run
- `--sizes 32 48 64 96 128 192` grid sizes with `nx==ny`
- `--nz 64` depth
- `--iters 128` iterations per run
- `--reps 50` repetitions per size (boxplots)
- `--halo 2` halo width
- `--threads N` set `OMP_NUM_THREADS`/`NUMBA_NUM_THREADS` for child processes
- `--numba-threading-layer {omp,workqueue,tbb,default}` choose Numba threading layer
- `--srun` wrap each run with `srun -n 1` (SLURM)
- `--env KEY=VAL` pass extra env to children (repeatable)
- `--extra "--plot_result"` append extra CLI to every child command

The plots are built accordingly:
- To estimate working set size in MB for the x-axis:
    - MB = nx * ny * nz * fields * bytes_per_scalar / 1e6
         = nx * ny * nz * 3      * 4                / 1e6
- y-axis -> runtime per gridpoint (µs) (the lower, the better)

# Post‑processing, comparing runtimes together — `overlay_and_speedup.py`

Creates two figures in the newest `bench_*` directory containing `*_raw.json` files (or pass a path explicitly).

- `overlay_medians.png` — overlay of median runtime/gridpoint vs working‑set MB (log‑log) for all frameworks found from the latest produced data via stencil_bench.py script or against one we select via terminal (example how to do so below).
- `speedup_vs_numpy.png` — speedup relative to NumPy at the common size across frameworks -> how much faster are the other methods.

Examples for terminal:
> python overlay_and_speedup.py               # auto‑select newest bench_* dir
> python overlay_and_speedup.py ./bench_17249 # or point to a specific directory

# Validation, precision & performance

- All drivers use float32 by default for fairness and lower memory bandwidth.
- The initial condition is a centered box (`=1` inside, `=0` outside), so diffusion smooths it over iterations.
- Very large grids can exhaust RAM/VRAM. The x‑axis in our plots is a quick estimate of working‑set size (3 fields × 4 bytes).

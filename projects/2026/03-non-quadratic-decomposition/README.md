# HPC4WC Project 3: Domain Decomposition SWE

Small 2D shallow-water-equation solver used to study domain decomposition and
MPI halo exchange. The project currently contains two main solver variants:

- `src/baseline.jl`: baseline implementation using `ImplicitGlobalGrid.jl`.
- `src/manual.jl`: manual MPI Cartesian-domain-decomposition version without
  `ImplicitGlobalGrid.jl`.

Both versions use `ParallelStencil` kernels. Both read the backend from the
`USE_GPU` environment variable:

- `manual.jl` defaults to the CPU backend (`USE_GPU` unset or `false`) and
  switches to CUDA with `USE_GPU=true`.
- `baseline.jl` defaults to CUDA (`USE_GPU=true`); set `USE_GPU=false` to run
  the CPU commands below.

## Provenance

The numerical solver is adapted from the SWE project reference implementation
`src/xpu/2d_swe_multi_xpu_wb.jl` in
[S1ntax3rror/ShallowWater4PDEonGPU](https://github.com/S1ntax3rror/ShallowWater4PDEonGPU).

This repository has been simplified for the HPC4WC domain-decomposition project:

- Uses one simple Gaussian free-surface perturbation as initial condition.
- Uses flat bathymetry (`z = 0`) only.
- Removes topography loading and related data preprocessing.
- Removes the expansion-factor/topography experiment code.
- Removes GPU memory-workaround/test code from the reference solver.
- Keeps array-output visualization instead of plotting from inside the solver.
- Adds benchmark mode that times only the main simulation loop.
- Adds saved domain-decomposition metadata for external plotting.
- Keeps communication simple: there is currently **no hidden/overlapped
  communication**. The manual version uses blocking halo exchange. Overlap can
  be added later as a separate optimization.

## Setup

Instantiate the Julia environment once:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

If the system `mpiexec` does not match the MPI used by `MPI.jl`, use:

```bash
julia --project -e 'using MPI; println(MPI.mpiexec())'
```

and launch through `MPI.mpiexec()` as shown below.

## Solver Options

Options shared by `src/baseline.jl` and `src/manual.jl`:

```text
--nx N          Global x size including outer halos/default boundary cells
--ny N          Global y size including outer halos/default boundary cells
--nt N          Number of timesteps
--outdir DIR    Directory for saved arrays and domain-decomposition metadata
--benchmark     Run timing mode: no saved arrays, no progress output
--warmup N      Warm-up iterations excluded from benchmark timing
--benchdir FILE CSV file to which benchmark results are appended
--test           Save the initial h state instead of time-stepping
--test-file FILE Output file for the --test initial-state snapshot
--viz            Save serialized field arrays for later plotting
```

Solver-specific options:

```text
manual.jl:   --topo "(PX,PY)"  MPI Cartesian topology (default "(2,2)")
manual.jl:   --runs N          Number of timed benchmark repetitions (default 10)
```

`baseline.jl` also accepts `--test`, `--test-file`, and `--viz`. It lets
`ImplicitGlobalGrid.jl`/MPI choose a compact Cartesian topology from the number
of MPI ranks. `manual.jl` accepts its topology with `--topo` and must use
`PX * PY` MPI ranks. In both cases, the global interior size `nx - 2` by
`ny - 2` must be divisible by the chosen topology.

## CPU Launches

Use the MPI launcher that belongs to `MPI.jl`; the setup command above prints
its path. The examples below use a small CPU-friendly grid.

`manual.jl` runs on the CPU by default. Setting `USE_GPU=false` explicitly makes
the intended backend clear:

```bash
USE_GPU=false mpiexec -n 4 julia -t 1 --project=. src/manual.jl \
    --topo "(2,2)" \
    --nx 1026 --ny 258 --nt 20 \
    --warmup 5 --runs 1 --benchmark \
    --benchdir output/cpu_manual.csv \
    --outdir output/cpu_manual
```

The topology product must equal the MPI process count; for example, `"(2,2)"`
requires four ranks.

### Baseline CPU launch

`baseline.jl` uses `ImplicitGlobalGrid.jl` for the Cartesian decomposition,
global indices, and halo updates. The topology is selected automatically from
the number of MPI ranks. To run on the CPU, force `USE_GPU=false`:

```bash
USE_GPU=false mpiexec -n 4 julia -t 1 --project=. src/baseline.jl \
    --nx 1026 --ny 258 --nt 20 \
    --warmup 5 --benchmark \
    --benchdir output/cpu_baseline.csv \
    --outdir output/cpu_baseline
```

Equivalent launch using the MPI selected by the Julia environment:

```bash
julia --project=. -e 'using MPI; run(addenv(`$(MPI.mpiexec()) -n 4 julia -t 1 --project=. src/baseline.jl --nx 1026 --ny 258 --nt 20 --warmup 5 --benchmark --benchdir output/cpu_baseline.csv --outdir output/cpu_baseline`, "USE_GPU"=>"false"))'
```

For a normal, non-benchmark baseline run, omit `--benchmark`. Add `--viz` to
save field arrays. A normal manual run also omits `--benchmark`; it currently
saves visualization data by default.

Benchmark CSV rows include the solver, process topology, global/local sizes,
runtime, steps per second, and cell updates per second. The manual solver writes
one row per `--runs` repetition.

## GPU / CSCS Launches

GPU launches, Slurm resource requests, CUDA-aware MPI variables, and CSCS uenv
settings are provided in the shell scripts under `run_files/`. In particular:

```bash
run_files/cluster_strong_scaling_16r.sh      # strong scaling, 16 ranks
run_files/cluster_manual_walltime_4r.sh      # manual wall-time sweep, 4 ranks
run_files/cluster_manual_walltime_16r.sh     # manual wall-time sweep, 16 ranks
```

Dedicated kernel- and halo-timing cluster scripts exist for both 4 and 16 ranks
(`cluster_manual_kernel_timing_*_r.sh`, `cluster_manual_halo_timing_*_r.sh`).

Inspect and adapt the relevant `.sh` file before submission because account,
paths, node counts, task counts, and uenv settings are cluster-specific. The
strong-scaling script is the canonical example of the current solver flags.

## Save And Plot Field Arrays

Use `--viz` to save serialized arrays in `--outdir`. The solver writes files
named `array_frame_*.jls`. Plot them afterwards with:

```bash
mpiexec -n 4 julia --project src/baseline.jl --nx 80 --ny 80 --nt 20 --viz --outdir docs/frames/baseline
julia --project scripts/visualize_arrays.jl --input docs/frames/baseline --output docs/frames/baseline/plots
```

The same workflow works for `src/manual.jl` (which saves frames by default in a
non-benchmark run):

```bash
USE_GPU=false mpiexec -n 4 julia --project=. src/manual.jl --topo "(2,2)" --nx 82 --ny 82 --nt 20 --outdir docs/frames/manual
julia --project scripts/visualize_arrays.jl --input docs/frames/manual --output docs/frames/manual/plots
```

`scripts/visualize_arrays.jl` options:

```text
--input DIR      Directory containing array_frame_*.jls
--output DIR     Directory where PNG plots are written
```

## Benchmark Data Provenance

The CSV files under `output/` (`walltimes_*.csv`, `kernel_timing_*.csv`,
`halo_exchange_*.csv`, `strong_scaling_manual.csv`) were produced by the benchmark
runs on CSCS Santis on the GPU backend (`USE_GPU=true`). The baseline strong-scaling
results are in `output/benchmark/strong_scaling_baseline.csv`. All are GPU results,
not local CPU runs. The local helper scripts (`run_files/local_*`) write their
output under `docs/` (frames, validation, and benchmarks) if you want to compare
with local CPU numbers.

## Plot Domain Decomposition

The benchmark CSV files (`output/walltimes_4_ranks.csv` and
`output/walltimes_16_ranks.csv`) contain the topology and global/local grid
sizes needed to reconstruct the rectangular domain decomposition.

Plot it with:

```bash
julia --project scripts/visualize_domain_decomposition.jl --input output/walltimes_16_ranks.csv --output output
```

`scripts/visualize_domain_decomposition.jl` options:

```text
--input PATH     A topology CSV or a directory containing such CSV files
--output PATH    Output directory or filename whose basename should be used
```

Note: the script scans for files whose name starts with `walltimes` when given
a directory, or accepts an explicit CSV path via `--input`. The plot is built
directly from the CSV's topology and global/local grid-size columns. It does not
load the large serialized domain arrays. Every run writes both
`domain_decompositions.png` and `domain_decompositions.pdf` (or the basename
given with `--output`).

## Timing Sources

Beyond the combined benchmark in `src/manual.jl`, three dedicated timing variants
isolate the two components of the iteration time:

- `src/manual_kernel_timing.jl`: times only the interior `ParallelStencil`
  kernels. Writes `kernel_walltime_seconds` per iteration to a CSV.
- `src/manual_halo_timing.jl`: times only the blocking MPI halo exchange.
  Writes `halo_walltime_seconds` per iteration to a CSV.
- `src/manual_profiling.jl`: adds NVTX range annotations for Nsight Systems
  GPU profiling. Used with `run_files/profile_manual.sh`.

All three take the same `--nx`, `--ny`, `--nt`, `--topo`, `--benchdir`, and
`USE_GPU` options as `manual.jl`. The kernel and halo variants are used together
with the analysis notebook `scripts/Benchmark_Analysis.ipynb` to separate compute
time from communication time.

## Local Helper Scripts (run_files)

Small Bash wrappers that launch the solvers with a fixed parameter set through
`MPI.mpiexec()` via `julia --project -e`. Adjust the variables at the top and
run from the project root (for example `bash run_files/local_manual_walltime_8r.sh`).

- `local_manual_walltime_8r.sh`: runs the manual solver in benchmark mode over
  several `--topo` topologies with 8 ranks. Appends to `docs/benchmark/`.
  Usable as a quick local benchmark.
- `local_run_benchmarks_halo.sh`: runs the halo-timing variant over topologies.
  Appends to `docs/benchmark/`.
- `local_test.sh`: runs `--test` for both `manual.jl` and `baseline.jl`, then
  compares the saved initial states with `scripts/compute_error.jl`. Writes
  snapshots to `docs/validation/`.
- `local_vizualizations.sh`: runs `--viz` for `manual.jl` and `baseline.jl` and
  plots the saved frames under `docs/frames/`.

Related cluster scripts (`cluster_manual_*_4r.sh`, `cluster_manual_*_16r.sh`,
`cluster_strong_scaling_16r.sh`) mirror these runs on CSCS with Slurm and the
`USE_GPU=true` backend.

## Verification Scripts (scripts)

- `compute_error.jl`: compares the last `.jls` state from two output
  directories (for example manual vs baseline) and prints the relative L2 error.
- `test_topologies.jl`: compares two specific `--test` snapshot files and prints
  the relative error. Accepts an optional topology label.
- `Benchmark_Analysis.ipynb`: notebook that loads the benchmark, kernel, halo,
  and strong-scaling CSVs and produces the report figures.

`--test` (in both `manual.jl` and `baseline.jl`) is an export mode: it
initializes the state, writes the initial water depth `h` to the file given by
`--test-file` (default `test_h.jls`), gathers it to the global grid, and exits
without time-stepping. It is meant for validation, to confirm that both solvers
produce the same initial state on a given grid.

## Implementation Notes

`baseline.jl`:

- Uses `init_global_grid(...)` from `ImplicitGlobalGrid.jl`.
- Uses IGG helpers such as global indexing and `update_halo!`.
- Gathers field arrays and domain-decomposition metadata for external plotting.

`manual.jl`:

- Creates an MPI Cartesian communicator with `MPI.Cart_create`.
- Uses `MPI.Cart_shift` to find neighbors.
- Uses `MPI.Gatherv!` to gather local interiors to rank 0.
- Uses blocking `MPI.Sendrecv!` halo updates.
- Applies physical boundary conditions only on ranks at global boundaries.

Future work:

- Add nonblocking halo exchange and overlap with interior computation.
- Add different sizes of localgrids

## References

1. [S1ntax3rror/ShallowWater4PDEonGPU](https://github.com/S1ntax3rror/ShallowWater4PDEonGPU)
2. [omlins/ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
3. [eth-cscs/ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl)

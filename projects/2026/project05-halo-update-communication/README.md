# Communication strategies for halo-updates

Project report for High Performance Computing for Weather and Climate, ETH Zurich

## Setup: Python Virtual Environment

Recreate the local `.venv` for this repo whenever you get a fresh checkout or
clone on Santis (compute node, e.g. a JupyterHub terminal, so `mpicc`/`mpif90`
from the `prgenv-gnu` uenv are on `PATH`):

```bash
./setup_venv.sh
```

This creates `.venv` in the repo root and installs the pinned packages from
`requirements.txt` (numpy, matplotlib, mpi4py, click, ipyparallel, ipcmagic).
`.venv` is gitignored, so it's safe to delete and recreate at any time:

```bash
rm -rf .venv && ./setup_venv.sh
```

Activate it in a terminal with:

```bash
source .venv/bin/activate
```

## Validation

The stencil implementations were validated against the serial reference using
this launcher (run from the repository root after building the executables):

```bash
uenv run prgenv-gnu/26.3:v1 --view=default -- ./validation/validate_all_strategies.sh
```

For each implementation, the launcher runs this comparison (`<strategy>` is the
implementation name):

```bash
./.venv/bin/python validation/compare_fields_f90.py \
  --src results/validation/fields/serial.dat \
  --trg "results/validation/fields/<strategy>.dat" \
  --rtol 1e-4 --atol 1e-4
```

## Experiments

Run from the repository root on Santis with the compiler environment and `.venv`
active. Build the executables, then run each sweep sequentially. New results go
into a separate directory under `results/`.

```bash
for version in comm_strats_timer comm_comp mpi mpi-corners mpi-corners-pipelined mpi-corners-waitany; do
  make VERSION="$version" || break
done
RUN_DIR="results/run_$(date +%Y%m%d_%H%M%S)"
```

### Blocking and Non-Blocking Communication

Weak scaling:

```bash
env STRATEGIES_FILTER=sendrecv,sendrecv_evenodd,brokencycles,irecvisend \
  ./experiment_scaling_all_strategies.sh --mode weak \
  --ranks 4,8,16,32,64 --local-n 128 --nz 64 --num_iter 1024 \
  --repeats 3 --results-dir "$RUN_DIR/blocking_non_blocking/weak"
```

Strong scaling:

```bash
env STRATEGIES_FILTER=sendrecv,sendrecv_evenodd,brokencycles,irecvisend \
  ./experiment_scaling_all_strategies.sh --mode strong \
  --ranks 4,8,16,32,64 --nx 1024 --ny 1024 --nz 64 \
  --num_iter 1024 --repeats 3 \
  --results-dir "$RUN_DIR/blocking_non_blocking/strong"
```

### Overlapping Computation and Communication

Strong scaling:

```bash
env STRATEGIES_FILTER=comm_comp,irecvisend \
  ./experiment_scaling_all_strategies.sh --mode strong \
  --ranks 1,2,4,8,16,32,64 --nx 1024 --ny 1024 --nz 64 \
  --num_iter 1024 --repeats 3 \
  --results-dir "$RUN_DIR/comm_comp/strong"
```

Weak scaling:

```bash
env STRATEGIES_FILTER=comm_comp,irecvisend \
  ./experiment_scaling_all_strategies.sh --mode weak \
  --ranks 1,2,4,8,16,32,64 --local-n 128 --nz 64 --num_iter 1024 \
  --repeats 3 --results-dir "$RUN_DIR/comm_comp/weak"
```

### Explicit Corner Exchange

Weak scaling:

```bash
env STRATEGIES_FILTER=corners,pipelined,waitany \
  ./experiment_scaling_all_strategies.sh --mode weak \
  --ranks 8,16,32,64,96 --local-n 128 --nz 64 --num_iter 1024 \
  --repeats 3 --results-dir "$RUN_DIR/corners/weak"

./experiment_mpi_leftright_scaling.sh --mode weak \
  --ranks 8,16,32,64,96 --local-n 128 --nz 64 --num_iter 1024 \
  --repeats 3 --results-dir "$RUN_DIR/corners/weak"
```

Strong scaling:

```bash
env STRATEGIES_FILTER=corners,pipelined,waitany \
  ./experiment_scaling_all_strategies.sh --mode strong \
  --ranks 8,16,32,64,96 --nx 1024 --ny 1024 --nz 64 \
  --num_iter 1024 --repeats 3 \
  --results-dir "$RUN_DIR/corners/strong"

./experiment_mpi_leftright_scaling.sh --mode strong \
  --ranks 8,16,32,64,96 --nx 1024 --ny 1024 --nz 64 \
  --num_iter 1024 --repeats 3 \
  --results-dir "$RUN_DIR/corners/strong"
```

## Plot Saved Results

### Blocking/non-blocking — weak scaling

```bash
python3 analyze_benchmark_results.py \
  --results-dir results/blocking_non_blocking/results_comm_strats_weak \
  --series-from strategy --mode weak --metric total_time \
  --csv-out plots/blocking_non_blocking/weak/comm_strategy_results.csv \
  --agg-csv-out plots/blocking_non_blocking/weak/comm_strategy_results_agg.csv \
  --out-dir plots/blocking_non_blocking/weak
```

### Blocking/non-blocking — strong scaling

```bash
python3 analyze_benchmark_results.py \
  --results-dir results/blocking_non_blocking/results_comm_strats_strong \
  --series-from strategy --mode strong --metric total_time \
  --csv-out plots/blocking_non_blocking/strong/comm_strategy_results.csv \
  --agg-csv-out plots/blocking_non_blocking/strong/comm_strategy_results_agg.csv \
  --out-dir plots/blocking_non_blocking/strong
```

### Computation/communication overlap — strong scaling

```bash
python3 analyze_benchmark_results.py \
  --results-dir results/comm_comp_experiments/strong_irecvisend \
  --series-from strategy --mode strong --metric total_time \
  --csv-out plots/comm_comp_experiments/strong_irecvisend/runs.csv \
  --agg-csv-out plots/comm_comp_experiments/strong_irecvisend/aggregated.csv \
  --out-dir plots/comm_comp_experiments/strong_irecvisend
```

### Corner exchange — weak scaling

```bash
python3 analyze_benchmark_results.py \
  --results-dir results/corners/results_weak_128_detailed_timing \
  --results-dir results/corners/results_weak_128_waitany_96_extra \
  --series-from strategy --mode weak --metric total_time \
  --csv-out plots/corners/plots_weak_128_detailed_timing/runs.csv \
  --agg-csv-out plots/corners/plots_weak_128_detailed_timing/aggregated.csv \
  --out-dir plots/corners/plots_weak_128_detailed_timing
```

### Corner exchange — strong scaling

```bash
python3 analyze_benchmark_results.py \
  --results-dir results/corners/results_strong_1024_detailed_timing \
  --results-dir results/corners/results_corners_strong_1024_timed_n96 \
  --results-dir results/corners/results_mpi_leftright_strong_1024_n96 \
  --series-from strategy --mode strong --metric total_time \
  --csv-out plots/corners/plots_strong_1024_detailed_timing/runs.csv \
  --agg-csv-out plots/corners/plots_strong_1024_detailed_timing/aggregated.csv \
  --out-dir plots/corners/plots_strong_1024_detailed_timing
```

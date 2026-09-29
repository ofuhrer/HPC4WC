# CNN Forward Pass: C++ vs PyTorch

HPC4WC course project, ETH Zurich, 2026. Matteo Meciani, Ari Hölscher, Nicola Rupf.

A convolutional neural network is trained on MNIST in PyTorch, then its forward pass
is reimplemented from scratch in C++ (multiple optimized variants: naive nested loops,
loop reordering, blocking, im2col + matmul, NEON intrinsics, ...). The C++ output is
verified bit-for-bit (up to floating-point precision) against PyTorch, and both are
benchmarked against each other on CSCS Santis.

## Repo structure

```
src/cpp/                       C++ forward-pass implementations
  cnn.cpp                        model definition mirroring model.py
  cnn_internals.cpp              layer kernels (conv2d, maxpool2d, linear, ...): the optimization target
  benchmark.cpp                  entry point: runs forward pass, verify, or profile modes
  include/                       headers (cnn.h, cnn_internals.h, timing.h, utils.h)

src/python/                    PyTorch reference implementation
  model.py                       network architecture
  train.py                       training script
  verify.py                      compares C++ output against PyTorch for numerical correctness
  benchmark.py                   PyTorch-side timing
  weights_cpp/                   exported weights, C++ logits, and timing/verification results

data/                          MNIST dataset — downloaded here on first run, git-ignored
scripts/                         Slurm submission scripts (run, verify, train)
build/                           compiled binary and generated assembly (.s) files
logs/                            Slurm job logs
Makefile                         build/run/submit targets, see `make help`
requirements.txt                 Python dependencies
hpc4wc_cnn_report.pdf            Project report
```

## Setup (CSCS Santis)

Done once per uenv version:

```bash
uenv image ls                                    # find latest pytorch/vX.Y.Z:vN
uenv start pytorch/vX.Y.Z:vN --view=default
unset PYTHONPATH
export PYTHONUSERBASE="$(dirname "$(dirname "$(which python)")")"
python -m venv --system-site-packages ~/venvs/venv
source ~/venvs/venv/bin/activate
pip install -r requirements.txt
deactivate
exit
```

Slurm jobs (`make run`, `make verify`, `make train`) activate the uenv and venv
automatically. Always run `make` from the repo root.

The MNIST dataset is **not committed**. It is fetched into `data/MNIST/raw/`
automatically the first time you run any `make run`/`verify`/`train` target
(local or Slurm) — the download runs on the login node, so it needs internet
there. To fetch it manually: `python3 src/python/download_data.py`.

## Usage

```bash
make help          # list all targets
```

Cluster (submits to Slurm via sbatch):

```bash
make run           # build C++ binary, run forward pass
make verify        # build C++ binary, run it, then verify against PyTorch
make train         # submit GPU training job
make all           # run + verify + train
make logs          # tail the latest Slurm log
```

Local (login node, no Slurm):

```bash
make build         # compile src/cpp -> build/cnn_forward
make run-local     # run the C++ binary, then benchmark.py
make verify-local  # run the C++ binary in verify mode, then verify.py
make train-local   # run train.py directly (DEVICE=cuda|cpu)
make all-local     # build -> run-local -> verify-local
```

Useful overrides:

```bash
make run NUM_RUNS=100 NUM_WARMUP_RUNS=20 BATCH_SIZE=64   # benchmark sizing
make DEBUG=1 build                                        # -O0 -g -fsanitize=address,undefined
make asm                                                  # emit annotated assembly to build/
make clean                                                # remove build artefacts and checkpoints
```

Manual compilation (no Makefile), from `src/cpp/`:

```bash
g++ -O3 -mcpu=native benchmark.cpp cnn.cpp cnn_internals.cpp -o cnn_forward
srun -A hpc4wc-course2026-ethz --partition=normal --time=00:05:00 --ntasks=1 ./cnn_forward
```

Profiling:

```bash
cd src/cpp && ../../build/cnn_forward profile
perf record -e cycles:u -g -o /tmp/perf.data -- ../../build/cnn_forward
perf report -i /tmp/perf.data
perf annotate -i /tmp/perf.data --stdio conv2d_forward
```

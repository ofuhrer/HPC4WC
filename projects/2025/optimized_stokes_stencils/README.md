# Optimized Matrix-free Smoothers for Variable-Viscosity Stokes Equations

>We present the design, implementation, and evaluation of optimized matrix-free stencil kernels for multigrid smoothing in the incom-
pressible Stokes equations with variable viscosity, motivated by geophysical flow problems. We investigate five smoother variants derived
from different optimisation strategies: Red–Black Gauss–Seidel, Jacobi, fused Jacobi, blocked fused Jacobi, and a novel Jacobi smoother
with RAS-type temporal blocking, a strategy that applies local iterations on overlapping tiles to improve cache reuse. To ensure correct-
ness, we introduce an energy-based residual norm that balances velocity and pressure contributions, and validate all implementations
using a high-contrast sinker benchmark representative of realistic geodynamic numerical models. Our performance study on NVIDIA
GH200 Grace Hopper nodes of the ALPS supercomputer demonstrates that all smoothers scale well within a single NUMA domain, but
the RAS-Jacobi smoother consistently achieves the best performance at higher core counts. It sustains over 90% weak-scaling efficiency
up to 64 cores and delivers up to a threefold speedup compared to the C++ Jacobi baseline, owing to improved cache reuse and reduced
memory traffic. These results show that temporal blocking, already employed in distributed-memory solvers to reduce communication,
can also provide substantial benefits at the socket and NUMA level. This work highlights the importance of cache-aware stencil design
for harnessing modern heterogeneous architectures and lays the groundwork for extending RAS-type temporal blocking strategies to
three-dimensional problems and GPU accelerators.

## Quick Start

### Prerequisites
To get started, ensure you have the following installed:

- Python: Version 3.8 or newer.
- gcc or clang compiler
- Python Packages: All necessary Python libraries are listed in `requirements.txt`.

### Installation

Follow these steps to set up the project:

```bash
# Clone the repository
git clone https://github.com/MarcelFerrari/HPC4WC.git
cd HPC4WC

# Create and activate a Python virtual environment
python -m venv .venv
source ./.venv/bin/activate

# Install required Python packages
pip install -r requirements.txt
```

## Reproduce Results

First, compile the C++ kernels. Ensure you are in the project's root directory before executing these commands.

```bash
# Prepare build target directory for CPU C++ code
mkdir ./CPU/cpp/build
cd ./CPU/cpp/build

# Build the C++ files using CMake
cmake ../CMakeLists.txt
make

# Return to the project root directory
cd ../../..
```

### Run Benchmarks

The benchmark.sh script allows you to run performance tests across all implemented configurations.

To see available options, use the --help flag:

```bash
bash ./benchmark.sh --help
```

For example, the following command runs the benchmark with:

- nx=ny=2000 (grid resolution)
- max_iter=10 (maximum iterations)
- Each configuration executed 1 time
- Measure for strong scaling, i.e. run with 1, 2, 4, ... threads

```bash
bash benchmark.sh 2000 10 1 strong
```

To capture the complete output of the benchmark run for later analysis, redirect it to a file:

```bash
bash benchmark.sh 2000 10 1 strong > output_[some-run-identifier].txt 2>&1
```

Finally, analyze the results offline using the provided Python script:

```bash
python analyze-benchmark-results.py output_[some-run-identifier].txt
```

## Authors

Marcel Ferrari,  Cyrill Püntener, Niklas Viebig  
July 2025
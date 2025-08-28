# HPC4WC Overlapping Performance Analysis

This project analyzes the performance benefits of using CUDA streams for overlapping computation and memory operations in high-performance computing applications. It includes implementations of diffusion stencil operations and arccos computations with varying levels of overlapping.

## Project Structure

```
HPC4WC-overlapping/
├── src/                    # Source code implementations
│   ├── cuda/              # CUDA implementations
│   │   ├── arccos_cuda.cu/.cuh      # Arccos kernel implementations
│   │   ├── stencil2d-gpu.cu         # Single-stream GPU stencil
│   │   ├── stencil2d-gpu_streams.cu # Multi-stream GPU stencil
│   │   ├── main.cu                  # Main arccos test program
│   │   ├── runtime_analysis.py      # Performance data analysis
│   │   └── stencil2d_helper/        # Helper utilities and base implementations
│   │       ├── stencil_kernels.cu/.cuh  # CUDA kernels for stencil operations
│   │       ├── stencil2d-base.cpp       # CPU baseline implementation
│   │       └── utils.h                  # Storage and utility classes
│   ├── gt4py/             # GT4Py reference implementations
│   │   ├── arccos_gt4py.py           # GT4Py arccos implementation
│   │   ├── run_arccos_gt4py.py       # Arccos GT4Py runner script
│   │   ├── run_stencil2d_gt4py.py    # Stencil GT4Py runner script
│   │   ├── stencil2d_gt4py.py        # GT4Py stencil implementation
│   │   ├── test_arccos_gt4py.py      # GT4Py arccos tests
│   │   └── test_timing_arccos.ipynb  # GT4Py timing analysis notebook
├── tests/                 # Test scripts and validation
│   ├── test_stencil.sh   # Automated stencil testing script
│   ├── comparison.py     # Output validation script
├── measurements/          # Performance data and analysis
│   ├── cuda_plots.ipynb  # Performance visualization notebooks
│   ├── arccos_cuda_vs_gt4py_plots.ipynb   # Performance comparison to GT4Py
│   ├── stencil_cuda_vs_gt4py_plots.ipynb  # Performance comparison to GT4Py
│   ├── *.csv             # Raw performance data
│   └── *.pdf             # Generated plots
├── report/               # Project documentation and reports
├── CMakeLists.txt        # Build configuration
├── environment.yml       # Conda environment specification
├── run_*.sh             # SLURM job scripts
└── README.md            # This file
```

## Quick Start

### 1. Environment Setup

First, create and activate the required Python environment:

```bash
# Clone the repository
git clone https://github.com/grafrap/HPC4WC-overlapping.git
cd HPC4WC-overlapping
```

Create conda environment.
Please follow the environment creation of the course under 
https://github.com/ofuhrer/HPC4WC/blob/main/setup/01-getting-started.pdf

Please make sure that the environment is activated and then install the required packages for this project

``` bash
pip install -r requirements.txt
```




### 2. Building the Code

The project uses CMake for building. On systems with CUDA:

```bash
# Create build directory
mkdir build
cd build

# Configure with CUDA (adjust CUDA path as needed and make sure that cmake finds nvcc)
cmake -DCUDAToolkit_ROOT=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3 ..

# if this cmake command fails, please use /usr/bin/cmake instead of cmake 

# Build all targets
make

# Available executables:
# - stencil2d_cpu         (CPU baseline)
# - stencil2d_gpu         (Single-stream GPU)
# - stencil2d_gpu_streams (Multi-stream GPU)
# - arccos_cuda           (Arccos performance tests)
```

### 3. Running Tests

The arccos_cuda program includes a validation step that can be manually configured through the last argument of the application (see "Usage").

Validate stencil implementations with the automated test suite:

```bash
cd tests
./test_stencil.sh
```

This script:
- Builds the CPU and GPU implementations
- Runs baseline CPU version
- Tests GPU versions with different stream counts
- Validates numerical correctness between implementations


## Usage

### Stencil Diffusion Operations

#### CPU Baseline
```bash
./stencil2d_cpu -nx 128 -ny 128 -nz 64 -iter 100
```

#### Single-Stream GPU
```bash
./stencil2d_gpu -nx 128 -ny 128 -nz 64 -iter 100
```

#### Multi-Stream GPU
```bash
./stencil2d_gpu_streams -nx 128 -ny 128 -nz 64 -iter 100 -streams 8 -test false
```

**Parameters:**
- `-nx`, `-ny`, `-nz`: Grid dimensions (excluding halo regions)
- `-iter`: Number of diffusion iterations
- `-streams`: Number of CUDA streams (GPU streams version only)
- `-test`: Enable test mode for output validation (GPU streams version only), this argument is optional

### Arccos Performance

```bash
./arccos_cuda 2 256 32 10 0
```
**Parameters**
./arccos_cuda <num_arccos_calls> <size> <num_streams> <num_repetitions> <validation_period>
- `num_arccos_calls`: Number of chained arccos operations on each element
- `size`: Size of the input array
- `num_streams`: Number of CUDA streams to use
- `num_repetitions`: Number of iterations to run for time measurement
- `validation_period`: Period in which results should be tested on correctness. 0 for no validaition, 1 for validation in each repetition.

Runs performance tests with given array sizes and stream count.

## Performance Analysis

### Running Benchmarks

Use the provided SLURM scripts for comprehensive performance analysis:

```bash
# Submit stencil performance jobs (from the main directory)
sbatch run-stencil_scaling.sh

# Submit arccos performance jobs  (from the main directory)
sbatch run-arccos_cuda.sh
```

### Analyzing Results

Performance data is automatically saved to the `measurements/` directory. Use the Jupyter notebooks for analysis.

## Implementation Details

### Diffusion Stencil

The project implements a 2D diffusion equation using a double Laplacian stencil:
```
out = in - α × ∇²(∇²(in))
```

Where:
- `∇²` is the 5-point Laplacian operator
- `α = 1/32` is the diffusion coefficient
- Periodic boundary conditions are applied

#### CUDA Kernels

- **`updateHaloKernel`**: Updates periodic boundary conditions
- **`diffusionStepKernel`**: Applies diffusion step with combined double Laplacian
- **`compute_kernel_multiple`**: Performs chained arccos operations

#### Memory Management

- Automatic device memory allocation/deallocation
- Asynchronous memory transfers when using streams
- Optimized memory access patterns for GPU performance

### Arc cosine

Model problem where arc cosine evaluations are performed on GPU. Overlapping of this computation and the memory transfer between host and device memory is achived by using multiple CUDA streams.

#### CUDA Kernels

- **`compute_kernel_multiple`**: Applies arc cosine operations

#### Memory Management

- Automatic device memory allocation/deallocation
- Asynchronous memory transfers

## File I/O - Stencil

Input and output fields are saved in binary format:
- `in_field.dat`: Initial field configuration
- `out_field.dat`: Final simulation result  
- `out_field_streams.dat`: Multi-stream result (test mode)

## Performance Metrics

### Stencil

Results are reported in CSV format with columns:
- Problem dimensions (nx, ny, nz)
- Number of iterations
- Number of streams (where applicable)
- Execution time in seconds

### Arc cosine

Here as well, results are reported in CSV format.
The columns are:
- Number of arc cosine operations per kernel call
- Size of the data array
- Number of streams
- Average runtime in seconds

## Requirements

### System Requirements
- NVIDIA GPU with CUDA capability
- CUDA Toolkit (version 12.3 or compatible)
- C++ compiler with C++11 support
- CMake 3.18+

### Python Environment
- Python 3.11+
- NumPy
- Pandas
- Matplotlib
- CuPy (CUDA 12.x) >= 13.4.1
- MPI4Py >= 4.1.0
- Jupyter >= 1.1.1
- IPykernel >= 6.29.5
- IPCmagic-CSCS >= 1.1.0
- Bash Kernel >= 0.10.0
- Traitlets >= 4.3.1
- IPython
- JupyterLab Launcher >= 0.5.0
- GT4Py (GridTools for Python)

## Troubleshooting

### Build Issues
- Ensure CUDA_ROOT is correctly set in build scripts
- Check that GPU compute capability is supported
- Verify CMake can find CUDA installation

### Runtime Issues
- Confirm sufficient GPU memory for problem size
- Check CUDA driver compatibility
- Validate input parameters are within reasonable ranges

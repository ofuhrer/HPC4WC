# Convolutional Neural Networks Inference in C++ for High Performance Scientific Computing

This project investigates the development and performance optimization of convolutional neural network (CNN) forward inference in C++. Follow the present README document for a step-by-step set of instructions to build and run the CPU and GPU binaries and to reproduce the experiments and results reported.

## CPU-Enabled Inference

This project provides two benchmarking scripts for the CPU implementations of the CNN:  
1. **Quick Benchmark** – lightweight, runs out of the box with no additional Python dependencies. It executes a short sweep over thread counts, prints timings directly to the console, and quickly reproduces the main scaling and performance trends shown in the report (no plots).  
2. **Full Benchmark** – more comprehensive, requires additional Python packages. It performs a denser sweep with more repeats, generates CSV summaries, and produces the figures included in the report. This script is suitable for in-depth analysis but takes several hours to run.  

### 1. Quick Benchmark

#### What it does
* Runs serial models at one thread and parallel models over a small thread sweep 
  Default threads: `[1, 2, 4, 8, 16]` with `10000` samples
* Uses a short warm-up per point, then repeats to reduce noise 
  Default repeats: `2`
* Reports total wall time and compute only time parsed from the binaries
* Saves results to a timestamped directory for easy comparison across run

#### Outputs
* Console log with per-run timings and medians
* CSV at `benchmarking/results/<timestamp>_quick_cpu/results.csv`  
  Columns: `model, threads, run_id, total_time_s, compute_time_s`
* Slurm log at `benchmarking/logs/quickbench_cpu-<jobid>.out` when submitted through the batch wrapper

#### How to run:

You only have to submit the wrapper script on Euler. This will load modules, build the CPU binaries, set threading environment variables, and run the benchmark.

Minimal example without flags:
```bash
sbatch benchmarking/quickbench_cpu.sh
```

Example with flags:
```bash
sbatch benchmarking/quickbench_cpu.sh \
  --repeats 1 \
  --samples 2000 \
  --threads 1 2 4 8 \
  --outdir benchmarking/results/quick_smoke_$(date +%Y%m%d_%H%M)
```

#### Notes
* The thread list is clamped to `SLURM_CPUS_PER_TASK`.
* `--outdir` overrides the default timestamped results directory.
* You can also run `benchmarking/quickbench_cpu.py` directly in an interactive shell after loading the necessary modules and building the binaries. The wrapper forwards the same flags and ensures a consistent environment.


### 2. Full Benchmark

#### What it does
* Runs serial models at one thread and parallel models over a denser thread sweep  
  Default threads: `[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]` with `50000` samples
* Uses a short warm up per point, then repeats for statistical robustness  
  Default repeats: `10`
* Parses total wall time, and compute only time
* Collects system and git metadata for reproducibility
* Saves raw and summary CSVs and produces figures for compute time with IQR, speedup, efficiency, and best end to end time

#### Outputs
* Console log with per run timings and a final summary of best end to end time per model
* Results at `benchmarking/results/<timestamp>_full_cpu/`, including  
  * `results.csv` with all runs  
  * `summary_compute.csv` and `summary_total.csv` with medians, interquartile ranges, speedup, efficiency  
  * `meta.json` with system and git information  
  * Plots: `time_compute_iqr.png`, `speedup_compute.png`, `efficiency_compute.png`, `best_total_per_model.png`
* Slurm log at `benchmarking/logs/fullbench_cpu-<jobid>.out` when submitted through the batch wrapper

#### How to run

**Requirements** 

The Python script requires `numpy`, `pandas`, and `matplotlib`. If these are not already available, create and activate a lightweight environment:
```bash
python3 -m venv --without-pip ~/cnn-bench
source ~/cnn-bench/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python /tmp/get-pip.py
pip install --upgrade pip
pip install numpy pandas matplotlib
```

After creating the environment just run the wrapper script. It loads modules, performs a clean rebuild of the CPU binaries, activates your Python environment, binds cores, and runs the benchmark.

Minimal example without flags:
```bash
sbatch benchmarking/fullbench_cpu.sh
```

Example with flags:
```bash
sbatch benchmarking/fullbench_cpu.sh --repeats 10 --samples 50000
```

If you did not create ~/cnn-bench, update the activation line in the batch script `benchmarking/fullbench_cpu.sh`:
```bash
source /path/to/your-env/bin/activate
```

#### Notes
* The thread list is clamped to `SLURM_CPUS_PER_TASK`.
* You can also run `benchmarking/fullbenchmark_cpu.py` directly in an interactive shell after loading the necessary modules, activating the environment and building the binaries.


## GPU-Enabled Inference

On Euler, have an active interactive/Jupyter allocation that has access to at least one GPU (or 8 GPUs to replicate the GPU benchmarking experiments in the report).

To ensure access to a GPU (at least), run the following:
```
which nvcc && nvidia-smi
```

You are well set if the output looks something similar to the following:
```
/cluster/software/stacks/2024-04/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-8.5.0/cuda-12.1.1-hnndc7mqnanii5nw47vmyj73pmrlqdie/bin/nvcc

+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.172.08             Driver Version: 570.172.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     On  |   00000000:01:00.0 Off |                  N/A |
| 27%   29C    P8             18W /  250W |       1MiB /  11264MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

```


### 1. Loading Necessary Modules & Building the Project

From the root project directory, make the module-loading script executable and source it to automatically load all required modules:

```bash
chmod +x runme.sh
source ./runme.sh
```

To build the project (GPU binary), run from the **project root**:



```bash
cmake -S . -B build-gpu \
  -DBUILD_GPU=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$LIBTORCH_ROOT" \
  -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
```

Then run:
```
cmake --build build-gpu --target test_mnist_gpu -j "$(nproc)"
```

This will generate the GPU executable binary in the `build-gpu/` directory.


### 2. Running GPU-Enabled Inference

From the **project root**:

```bash
./build-gpu/test_mnist_gpu
```

This runs inference using the GPU-accelerated CNN model on the MNIST test set. The model processes the dataset in chunks. The final line summarizes the total number of samples processed and the corresponding wall-clock time.
```
...
Sample 9998: predicted=2, true=0
Sample 9999: predicted=7, true=0
Rank 0: processed 10000 samples in 1.148 seconds.
```

### 3. Benchmarking

For this section, you need to have access to multipe GPUs (e.g. 8 GPUs). From the **project root**, run:

```bash
bash benchmarking/run_gpu_benchmark.sh
```

This will run inference using different numbers of GPUs and generate two benchmarking files:

* `logs/results_strong.tsv`
* `logs/results_weak.tsv`

Each contains the timing, speedup, and efficiency results for strong and weak scaling tests respectively.


### 4. Plotting Results

First, set up a virtual environment and install dependencies

```bash
module load stack/2025-06 gcc/8.5.0 python/3.11.9 texlive/20240312  
python3 -m venv venv                
source venv/bin/activate            
pip install pandas matplotlib       
```

Then, from the **project root**, run:

```bash
python plot_gpu_scaling.py
```

This generates the corresponding benchmarking results as PNG plots in the `figs/GPU` folder.

Similarly, to run xPU benchmarking comparing the scaling performance across all the parallelized versions in this project, run the following:
```
python plot_xpu_cross_scaling.py
```
This benchmarking plots are now saved as PNG plots in the `figs/` folder.

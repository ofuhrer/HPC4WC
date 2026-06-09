# hpc4wc:student-begin
# hpc4wc:student | # Phind ai-assistant (405B; 06-2025) was used for the development of this Python script.
# hpc4wc:student | # Especially the different meanings of memory management types and how they relate to CuPy-settings was prompted.
# hpc4wc:student | # The code has been adapted and reviewed before publication. The verification of the results was carried out
# hpc4wc:student | # using visual aids and summary statistics. An in-depth analysis with an expert from CSCS/NVIDIA/CuPy is still outstanding.
# hpc4wc:student |
# hpc4wc:student | import numpy as np
# hpc4wc:student | import cupy as cp
# hpc4wc:student | import matplotlib.pyplot as plt
# hpc4wc:student | import time
# hpc4wc:student | import gc
# hpc4wc:student |
# hpc4wc:student | def warmup_cupy():
# hpc4wc:student |     """Perform initial CuPy operations to initialize CUDA context and JIT compilation."""
# hpc4wc:student |     print("Warming up CuPy and CUDA runtime...")
# hpc4wc:student |
# hpc4wc:student |     # Initialize CUDA context and trigger JIT compilation
# hpc4wc:student |     a = cp.array([1, 2, 3])
# hpc4wc:student |     b = cp.array([4, 5, 6])
# hpc4wc:student |     cp.add(a, b)
# hpc4wc:student |     cp.multiply(a, b)
# hpc4wc:student |     cp.sum(a)
# hpc4wc:student |
# hpc4wc:student |     # Trigger random number generation
# hpc4wc:student |     cp.random.random((100, 100))
# hpc4wc:student |
# hpc4wc:student |     # Force synchronization
# hpc4wc:student |     cp.cuda.Device().synchronize()
# hpc4wc:student |
# hpc4wc:student |     # Print device information
# hpc4wc:student |     device_id = cp.cuda.Device().id
# hpc4wc:student |     try:
# hpc4wc:student |         device_props = cp.cuda.runtime.getDeviceProperties(device_id)
# hpc4wc:student |         print(f"CUDA Device ID: {device_id}")
# hpc4wc:student |         print(f"CUDA Device Name: {device_props['name'].decode()}")
# hpc4wc:student |     except:
# hpc4wc:student |         print(f"CUDA Device ID: {device_id}")
# hpc4wc:student |         print("CUDA Device Name: Unable to determine")
# hpc4wc:student |
# hpc4wc:student |     print(f"CuPy Version: {cp.__version__}")
# hpc4wc:student |     print("-" * 60)
# hpc4wc:student |
# hpc4wc:student | def clear_memory_pools():
# hpc4wc:student |     """Clear both device and pinned memory pools."""
# hpc4wc:student |     mempool = cp.get_default_memory_pool()
# hpc4wc:student |     pinned_mempool = cp.get_default_pinned_memory_pool()
# hpc4wc:student |
# hpc4wc:student |     mempool.free_all_blocks()
# hpc4wc:student |     pinned_mempool.free_all_blocks()
# hpc4wc:student |
# hpc4wc:student |     # Force garbage collection
# hpc4wc:student |     gc.collect()
# hpc4wc:student |
# hpc4wc:student | class AtmosphericModel:
# hpc4wc:student |     def __init__(self, nx=128, ny=128, nz=64, memory_type='device'):
# hpc4wc:student |         """Initialize atmospheric model with specified memory type."""
# hpc4wc:student |         self.nx, self.ny, self.nz = nx, ny, nz
# hpc4wc:student |         self.memory_type = memory_type
# hpc4wc:student |         self.shape = (nz, ny, nx)
# hpc4wc:student |
# hpc4wc:student |         # Calculate exact size in bytes (float32 = 4 bytes)
# hpc4wc:student |         self.size_bytes = nx * ny * nz * 4
# hpc4wc:student |         self.size_mb = self.size_bytes / (1024 * 1024)
# hpc4wc:student |         self.size_gb = self.size_mb / 1024
# hpc4wc:student |
# hpc4wc:student |         # Initialize arrays based on memory type
# hpc4wc:student |         if memory_type == 'device':
# hpc4wc:student |             # Standard device memory allocation
# hpc4wc:student |             self.temperature = cp.random.random(self.shape, dtype=cp.float32)
# hpc4wc:student |
# hpc4wc:student |         elif memory_type == 'system':
# hpc4wc:student |             # Create on CPU and transfer to GPU
# hpc4wc:student |             temp_cpu = np.random.rand(*self.shape).astype(np.float32)
# hpc4wc:student |             self.temperature = cp.asarray(temp_cpu)
# hpc4wc:student |
# hpc4wc:student |         elif memory_type == 'managed':
# hpc4wc:student |             # Allocate true CUDA Unified Memory through CuPy's managed allocator.
# hpc4wc:student |             self.managed_mempool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
# hpc4wc:student |             with cp.cuda.using_allocator(self.managed_mempool.malloc):
# hpc4wc:student |                 self.temperature = cp.random.random(self.shape, dtype=cp.float32)
# hpc4wc:student |
# hpc4wc:student |         elif memory_type == 'pinned':
# hpc4wc:student |             # Use pinned memory for host array
# hpc4wc:student |             # Allocate pinned memory using CuPy's memory pool
# hpc4wc:student |             mem_size = self.size_bytes
# hpc4wc:student |             temp_mem = cp.cuda.alloc_pinned_memory(mem_size)
# hpc4wc:student |             temp_cpu = np.frombuffer(temp_mem, dtype=np.float32, count=self.nx*self.ny*self.nz).reshape(self.shape)
# hpc4wc:student |             temp_cpu[:] = np.random.rand(*self.shape)
# hpc4wc:student |             self.temperature_cpu = temp_cpu
# hpc4wc:student |             self.temperature = cp.asarray(self.temperature_cpu)
# hpc4wc:student |
# hpc4wc:student |         # Ensure all initialization is complete
# hpc4wc:student |         cp.cuda.Device().synchronize()
# hpc4wc:student |
# hpc4wc:student |     def get_cpu_data(self, n_repeat=1):
# hpc4wc:student |         """
# hpc4wc:student |         Transfer temperature data from GPU to CPU multiple times and return average time.
# hpc4wc:student |         Uses simple but reliable timing approach.
# hpc4wc:student |         """
# hpc4wc:student |         # Warm-up transfers to ensure JIT compilation is complete
# hpc4wc:student |         for _ in range(3):
# hpc4wc:student |             _ = self.temperature.get()
# hpc4wc:student |             cp.cuda.Device().synchronize()
# hpc4wc:student |
# hpc4wc:student |         # Measure multiple transfers
# hpc4wc:student |         times = []
# hpc4wc:student |         for i in range(n_repeat):
# hpc4wc:student |             # Force garbage collection to minimize interference
# hpc4wc:student |             gc.collect()
# hpc4wc:student |
# hpc4wc:student |             # Ensure GPU is idle before starting measurement
# hpc4wc:student |             cp.cuda.Device().synchronize()
# hpc4wc:student |
# hpc4wc:student |             # Start timing
# hpc4wc:student |             start_time = time.perf_counter()
# hpc4wc:student |
# hpc4wc:student |             # Perform the transfer
# hpc4wc:student |             result = self.temperature.get()
# hpc4wc:student |
# hpc4wc:student |             # Ensure transfer is complete
# hpc4wc:student |             cp.cuda.Device().synchronize()
# hpc4wc:student |
# hpc4wc:student |             # End timing
# hpc4wc:student |             end_time = time.perf_counter()
# hpc4wc:student |
# hpc4wc:student |             # Calculate elapsed time
# hpc4wc:student |             elapsed_sec = end_time - start_time
# hpc4wc:student |             times.append(elapsed_sec)
# hpc4wc:student |
# hpc4wc:student |         # Calculate statistics
# hpc4wc:student |         avg_time = sum(times) / len(times)
# hpc4wc:student |         min_time = min(times)
# hpc4wc:student |         max_time = max(times)
# hpc4wc:student |
# hpc4wc:student |         # Calculate bandwidth
# hpc4wc:student |         avg_bandwidth = self.size_gb / avg_time if avg_time > 0 else 0
# hpc4wc:student |         max_bandwidth = self.size_gb / min_time if min_time > 0 else 0
# hpc4wc:student |
# hpc4wc:student |         return result, avg_time, avg_bandwidth
# hpc4wc:student |
# hpc4wc:student | def run_benchmark():
# hpc4wc:student |     """Run the full benchmark and return results."""
# hpc4wc:student |     # Initialize CUDA context and JIT compilation before benchmarking
# hpc4wc:student |     warmup_cupy()
# hpc4wc:student |
# hpc4wc:student |     # Benchmark configuration - reduced grid sizes to avoid crashes
# hpc4wc:student |     grid_sizes = [128, 256, 512, 1024]
# hpc4wc:student |     memory_types = ['device', 'system', 'managed', 'pinned']
# hpc4wc:student |     transfer_times = {mem_type: [] for mem_type in memory_types}
# hpc4wc:student |     bandwidths = {mem_type: [] for mem_type in memory_types}
# hpc4wc:student |     array_sizes_mb = []
# hpc4wc:student |     array_sizes_gb = []
# hpc4wc:student |
# hpc4wc:student |     # Calculate number of repeats needed for each grid size
# hpc4wc:student |     repeats = {
# hpc4wc:student |         128: 32,
# hpc4wc:student |         256: 16,
# hpc4wc:student |         512: 8,
# hpc4wc:student |         1024: 4
# hpc4wc:student |     }
# hpc4wc:student |
# hpc4wc:student |     # Run benchmarks
# hpc4wc:student |     for nx in grid_sizes:
# hpc4wc:student |         ny = nx
# hpc4wc:student |         nz = 4
# hpc4wc:student |         print(f"Benchmarking grid size: {nx}x{ny}x{nz}")
# hpc4wc:student |
# hpc4wc:student |         # Calculate array size
# hpc4wc:student |         size_bytes = nx * ny * nz * 4  # float32 = 4 bytes
# hpc4wc:student |         size_mb = size_bytes / (1024 * 1024)
# hpc4wc:student |         size_gb = size_mb / 1024
# hpc4wc:student |         array_sizes_mb.append(size_mb)
# hpc4wc:student |         array_sizes_gb.append(size_gb)
# hpc4wc:student |
# hpc4wc:student |         for mem_type in memory_types:
# hpc4wc:student |             print(f"  Testing {mem_type} memory...")
# hpc4wc:student |
# hpc4wc:student |             try:
# hpc4wc:student |                 # Clear memory pools before each test
# hpc4wc:student |                 clear_memory_pools()
# hpc4wc:student |
# hpc4wc:student |                 # Create model
# hpc4wc:student |                 model = AtmosphericModel(nx=nx, ny=ny, nz=nz, memory_type=mem_type)
# hpc4wc:student |
# hpc4wc:student |                 n_repeat = repeats[nx]
# hpc4wc:student |
# hpc4wc:student |                 # Measure GPU->CPU transfer time with multiple repetitions
# hpc4wc:student |                 _, t_time, bandwidth = model.get_cpu_data(n_repeat=n_repeat)
# hpc4wc:student |                 transfer_times[mem_type].append(t_time)
# hpc4wc:student |                 bandwidths[mem_type].append(bandwidth)
# hpc4wc:student |
# hpc4wc:student |                 # Clean up GPU memory
# hpc4wc:student |                 del model
# hpc4wc:student |
# hpc4wc:student |                 # Clear memory pools after each test
# hpc4wc:student |                 clear_memory_pools()
# hpc4wc:student |
# hpc4wc:student |             except Exception as e:
# hpc4wc:student |                 print(f"    ERROR: {e}")
# hpc4wc:student |                 print(f"    Skipping this test and recording zero bandwidth")
# hpc4wc:student |                 transfer_times[mem_type].append(0)
# hpc4wc:student |                 bandwidths[mem_type].append(0)
# hpc4wc:student |
# hpc4wc:student |                 # Try to clean up
# hpc4wc:student |                 try:
# hpc4wc:student |                     del model
# hpc4wc:student |                 except:
# hpc4wc:student |                     pass
# hpc4wc:student |
# hpc4wc:student |                 # Force memory cleanup
# hpc4wc:student |                 clear_memory_pools()
# hpc4wc:student |
# hpc4wc:student |             # Small delay between tests
# hpc4wc:student |             time.sleep(1)
# hpc4wc:student |
# hpc4wc:student |     return grid_sizes, array_sizes_mb, array_sizes_gb, transfer_times, bandwidths
# hpc4wc:student | def plot_results(grid_sizes, array_sizes_mb, bandwidths, memory_types):
# hpc4wc:student |     """Plot the benchmark results with simplified formatting."""
# hpc4wc:student |     plt.figure(figsize=(12, 8))
# hpc4wc:student |
# hpc4wc:student |     # Plot each memory type with default styling
# hpc4wc:student |     for mem_type in memory_types:
# hpc4wc:student |         # Get the bandwidth data for this memory type
# hpc4wc:student |         bw_data = bandwidths[mem_type]
# hpc4wc:student |
# hpc4wc:student |         # Plot the data with default styling
# hpc4wc:student |         plt.plot(grid_sizes, bw_data,
# hpc4wc:student |                  marker='o',  # Keep just a simple marker
# hpc4wc:student |                  linewidth=2,
# hpc4wc:student |                  label=f"{mem_type} memory")
# hpc4wc:student |
# hpc4wc:student |     # Add labels and title
# hpc4wc:student |     plt.xlabel('Grid Size (N for NxNx4)', fontsize=14)
# hpc4wc:student |     plt.ylabel('Transfer Bandwidth (GB/s)', fontsize=14)
# hpc4wc:student |     plt.title('GPU-to-CPU Data Transfer Bandwidth vs Grid Size', fontsize=16)
# hpc4wc:student |
# hpc4wc:student |     # Add grid and legend
# hpc4wc:student |     plt.grid(True, which="both", ls="--", alpha=0.7)
# hpc4wc:student |     plt.legend(fontsize=12)
# hpc4wc:student |
# hpc4wc:student |     # Set x-ticks to show the actual grid sizes
# hpc4wc:student |     plt.xticks(grid_sizes, [str(size) for size in grid_sizes])
# hpc4wc:student |
# hpc4wc:student |     plt.tight_layout()
# hpc4wc:student |     plt.savefig('gpu_transfer_bandwidth.png', dpi=300)
# hpc4wc:student |     plt.show()
# hpc4wc:student |
# hpc4wc:student | def print_summary(grid_sizes, array_sizes_mb, array_sizes_gb, bandwidths, memory_types):
# hpc4wc:student |     """Print a summary table of the benchmark results."""
# hpc4wc:student |     print("\nSummary of Transfer Bandwidths (GB/s):")
# hpc4wc:student |     print("-" * 80)
# hpc4wc:student |     print(f"{'Grid Size':<15} | {'Array Size (MB)':<15} | " + " | ".join(f"{mem_type:<10}" for mem_type in memory_types))
# hpc4wc:student |     print("-" * 80)
# hpc4wc:student |
# hpc4wc:student |     for i, nx in enumerate(grid_sizes):
# hpc4wc:student |         size_mb = array_sizes_mb[i]
# hpc4wc:student |         bandwidths_str = " | ".join(f"{bandwidths[mem_type][i]:<10.2f}" for mem_type in memory_types)
# hpc4wc:student |         print(f"{f'{nx}×{nx}×4':<15} | {size_mb:<15.2f} | {bandwidths_str}")
# hpc4wc:student |
# hpc4wc:student | def main():
# hpc4wc:student |     """Main function to run the benchmark."""
# hpc4wc:student |     grid_sizes, array_sizes_mb, array_sizes_gb, transfer_times, bandwidths = run_benchmark()
# hpc4wc:student |     memory_types = ['device', 'system', 'managed', 'pinned']
# hpc4wc:student |
# hpc4wc:student |     # Plot results
# hpc4wc:student |     plot_results(grid_sizes, array_sizes_mb, bandwidths, memory_types)
# hpc4wc:student |
# hpc4wc:student |     # Print summary
# hpc4wc:student |     print_summary(grid_sizes, array_sizes_mb, array_sizes_gb, bandwidths, memory_types)
# hpc4wc:student |
# hpc4wc:student | if __name__ == "__main__":
# hpc4wc:student |     main()
# hpc4wc:student-end
# hpc4wc:solution-begin
# Phind ai-assistant (405B; 06-2025) was used for the development of this Python script.
# Especially the different meanings of memory management types and how they relate to CuPy-settings was prompted.
# The code has been adapted and reviewed before publication. The verification of the results was carried out
# using visual aids and summary statistics. An in-depth analysis with an expert from CSCS/NVIDIA/CuPy is still outstanding.

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
import gc

def warmup_cupy():
    """Perform initial CuPy operations to initialize CUDA context and JIT compilation."""
    print("Warming up CuPy and CUDA runtime...")

    # Initialize CUDA context and trigger JIT compilation
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5, 6])
    cp.add(a, b)
    cp.multiply(a, b)
    cp.sum(a)

    # Trigger random number generation
    cp.random.random((100, 100))

    # Force synchronization
    cp.cuda.Device().synchronize()

    # Print device information
    device_id = cp.cuda.Device().id
    try:
        device_props = cp.cuda.runtime.getDeviceProperties(device_id)
        print(f"CUDA Device ID: {device_id}")
        print(f"CUDA Device Name: {device_props['name'].decode()}")
    except:
        print(f"CUDA Device ID: {device_id}")
        print("CUDA Device Name: Unable to determine")

    print(f"CuPy Version: {cp.__version__}")
    print("-" * 60)

def clear_memory_pools():
    """Clear both device and pinned memory pools."""
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    # Force garbage collection
    gc.collect()

class AtmosphericModel:
    def __init__(self, nx=128, ny=128, nz=64, memory_type='device'):
        """Initialize atmospheric model with specified memory type."""
        self.nx, self.ny, self.nz = nx, ny, nz
        self.memory_type = memory_type
        self.shape = (nz, ny, nx)

        # Calculate exact size in bytes (float32 = 4 bytes)
        self.size_bytes = nx * ny * nz * 4
        self.size_mb = self.size_bytes / (1024 * 1024)
        self.size_gb = self.size_mb / 1024

        # Initialize arrays based on memory type
        if memory_type == 'device':
            # Standard device memory allocation
            self.temperature = cp.random.random(self.shape, dtype=cp.float32)

        elif memory_type == 'system':
            # Create on CPU and transfer to GPU
            temp_cpu = np.random.rand(*self.shape).astype(np.float32)
            self.temperature = cp.asarray(temp_cpu)

        elif memory_type == 'managed':
            # Allocate true CUDA Unified Memory through CuPy's managed allocator.
            self.managed_mempool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
            with cp.cuda.using_allocator(self.managed_mempool.malloc):
                self.temperature = cp.random.random(self.shape, dtype=cp.float32)

        elif memory_type == 'pinned':
            # Use pinned memory for host array
            # Allocate pinned memory using CuPy's memory pool
            mem_size = self.size_bytes
            temp_mem = cp.cuda.alloc_pinned_memory(mem_size)
            temp_cpu = np.frombuffer(temp_mem, dtype=np.float32, count=self.nx*self.ny*self.nz).reshape(self.shape)
            temp_cpu[:] = np.random.rand(*self.shape)
            self.temperature_cpu = temp_cpu
            self.temperature = cp.asarray(self.temperature_cpu)

        # Ensure all initialization is complete
        cp.cuda.Device().synchronize()

    def get_cpu_data(self, n_repeat=1):
        """
        Transfer temperature data from GPU to CPU multiple times and return average time.
        Uses simple but reliable timing approach.
        """
        # Warm-up transfers to ensure JIT compilation is complete
        for _ in range(3):
            _ = self.temperature.get()
            cp.cuda.Device().synchronize()

        # Measure multiple transfers
        times = []
        for i in range(n_repeat):
            # Force garbage collection to minimize interference
            gc.collect()

            # Ensure GPU is idle before starting measurement
            cp.cuda.Device().synchronize()

            # Start timing
            start_time = time.perf_counter()

            # Perform the transfer
            result = self.temperature.get()

            # Ensure transfer is complete
            cp.cuda.Device().synchronize()

            # End timing
            end_time = time.perf_counter()

            # Calculate elapsed time
            elapsed_sec = end_time - start_time
            times.append(elapsed_sec)

        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        # Calculate bandwidth
        avg_bandwidth = self.size_gb / avg_time if avg_time > 0 else 0
        max_bandwidth = self.size_gb / min_time if min_time > 0 else 0

        return result, avg_time, avg_bandwidth

def run_benchmark():
    """Run the full benchmark and return results."""
    # Initialize CUDA context and JIT compilation before benchmarking
    warmup_cupy()

    # Benchmark configuration - reduced grid sizes to avoid crashes
    grid_sizes = [128, 256, 512, 1024]
    memory_types = ['device', 'system', 'managed', 'pinned']
    transfer_times = {mem_type: [] for mem_type in memory_types}
    bandwidths = {mem_type: [] for mem_type in memory_types}
    array_sizes_mb = []
    array_sizes_gb = []

    # Calculate number of repeats needed for each grid size
    repeats = {
        128: 32,
        256: 16,
        512: 8,
        1024: 4
    }

    # Run benchmarks
    for nx in grid_sizes:
        ny = nx
        nz = 4
        print(f"Benchmarking grid size: {nx}x{ny}x{nz}")

        # Calculate array size
        size_bytes = nx * ny * nz * 4  # float32 = 4 bytes
        size_mb = size_bytes / (1024 * 1024)
        size_gb = size_mb / 1024
        array_sizes_mb.append(size_mb)
        array_sizes_gb.append(size_gb)

        for mem_type in memory_types:
            print(f"  Testing {mem_type} memory...")

            try:
                # Clear memory pools before each test
                clear_memory_pools()

                # Create model
                model = AtmosphericModel(nx=nx, ny=ny, nz=nz, memory_type=mem_type)

                n_repeat = repeats[nx]

                # Measure GPU->CPU transfer time with multiple repetitions
                _, t_time, bandwidth = model.get_cpu_data(n_repeat=n_repeat)
                transfer_times[mem_type].append(t_time)
                bandwidths[mem_type].append(bandwidth)

                # Clean up GPU memory
                del model

                # Clear memory pools after each test
                clear_memory_pools()

            except Exception as e:
                print(f"    ERROR: {e}")
                print(f"    Skipping this test and recording zero bandwidth")
                transfer_times[mem_type].append(0)
                bandwidths[mem_type].append(0)

                # Try to clean up
                try:
                    del model
                except:
                    pass

                # Force memory cleanup
                clear_memory_pools()

            # Small delay between tests
            time.sleep(1)

    return grid_sizes, array_sizes_mb, array_sizes_gb, transfer_times, bandwidths
def plot_results(grid_sizes, array_sizes_mb, bandwidths, memory_types):
    """Plot the benchmark results with simplified formatting."""
    plt.figure(figsize=(12, 8))

    # Plot each memory type with default styling
    for mem_type in memory_types:
        # Get the bandwidth data for this memory type
        bw_data = bandwidths[mem_type]

        # Plot the data with default styling
        plt.plot(grid_sizes, bw_data,
                 marker='o',  # Keep just a simple marker
                 linewidth=2,
                 label=f"{mem_type} memory")

    # Add labels and title
    plt.xlabel('Grid Size (N for NxNx4)', fontsize=14)
    plt.ylabel('Transfer Bandwidth (GB/s)', fontsize=14)
    plt.title('GPU-to-CPU Data Transfer Bandwidth vs Grid Size', fontsize=16)

    # Add grid and legend
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=12)

    # Set x-ticks to show the actual grid sizes
    plt.xticks(grid_sizes, [str(size) for size in grid_sizes])

    plt.tight_layout()
    plt.savefig('gpu_transfer_bandwidth.png', dpi=300)
    plt.show()

def print_summary(grid_sizes, array_sizes_mb, array_sizes_gb, bandwidths, memory_types):
    """Print a summary table of the benchmark results."""
    print("\nSummary of Transfer Bandwidths (GB/s):")
    print("-" * 80)
    print(f"{'Grid Size':<15} | {'Array Size (MB)':<15} | " + " | ".join(f"{mem_type:<10}" for mem_type in memory_types))
    print("-" * 80)

    for i, nx in enumerate(grid_sizes):
        size_mb = array_sizes_mb[i]
        bandwidths_str = " | ".join(f"{bandwidths[mem_type][i]:<10.2f}" for mem_type in memory_types)
        print(f"{f'{nx}×{nx}×4':<15} | {size_mb:<15.2f} | {bandwidths_str}")

def main():
    """Main function to run the benchmark."""
    grid_sizes, array_sizes_mb, array_sizes_gb, transfer_times, bandwidths = run_benchmark()
    memory_types = ['device', 'system', 'managed', 'pinned']

    # Plot results
    plot_results(grid_sizes, array_sizes_mb, bandwidths, memory_types)

    # Print summary
    print_summary(grid_sizes, array_sizes_mb, array_sizes_gb, bandwidths, memory_types)

if __name__ == "__main__":
    main()
# hpc4wc:solution-end

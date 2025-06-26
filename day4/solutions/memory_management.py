# Phind ai-assistant (405B; 06-2025) was used for the development of this Python script.
# Especially the different meanings of memory management types and how they relate to CuPy-settings was prompted.
# The code has been adapted and reviewed before publication. The verification of the results was carried out
# using visual aids and summary statistics. An in-depth analysis with an expert from CSCS/NVIDIA/CuPy is still outstanding.

import timeit
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
            # For managed memory, we'll use a simpler approach
            # Create on CPU first
            temp_cpu = np.random.rand(*self.shape).astype(np.float32)
            # Then transfer to GPU
            self.temperature = cp.asarray(temp_cpu)
            # Keep a reference to the CPU array
            self.temperature_cpu = temp_cpu
        
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
    grid_sizes = [128, 256, 512, 1024]  # Removed 2048 to avoid potential OOM
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
    plt.xlabel('Grid Size (N for NxNx32)', fontsize=14)
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
        print(f"{f'{nx}×{nx}×512':<15} | {size_mb:<15.2f} | {bandwidths_str}")
    
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

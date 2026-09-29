#!/bin/bash
#SBATCH --job-name=stencil_gpu_scaling
#SBATCH --output=measurements/stencil_scaling_%j.out
#SBATCH --error=measurements/stencil_scaling_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --exclusive

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Set up CUDA environment
export CUDA_ROOT=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3
export PATH=${CUDA_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:$LD_LIBRARY_PATH

# Build the project
/usr/bin/cmake -DCUDAToolkit_ROOT=${CUDA_ROOT} .. > /dev/null 2>&1
make > /dev/null 2>&1

# Create output file for results
OUTPUT_FILE="../measurements/stencil_gpu_scaling_$SLURM_JOB_ID.out"
ERROR_FILE="../measurements/stencil_gpu_scaling_$SLURM_JOB_ID.err"
mkdir -p ../measurements

# Print header only once
echo "ranks, nx, ny, nz, num_iter, time, num_streams" > $OUTPUT_FILE

# Four-dimensional loop for comprehensive testing
# Loop 1: Number of streams (2^0 to 2^9 = 1 to 512)
for z_size in 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536; do
    
    # Loop 2: Problem sizes (xy dimensions, powers of 2 from 2^3 to 2^15)
    for ((xy_exp=7; xy_exp<=7; xy_exp++)); do
        XY_SIZE=$((2**xy_exp))
        
        # Loop 3: Z dimension (8 to 512, exponential steps)
            # Loop 4: Iteration counts (2^0 to 2^15 = 1 to 32768)
            for ((iter_exp=5; iter_exp<=10; iter_exp++)); do
                iter=$((2**iter_exp))
                for ((stream_exp=0; stream_exp<=8; stream_exp++)); do
                    streams=$((2**stream_exp))
                
                # Run the test and append results
                ./stencil2d_gpu_streams -nx $XY_SIZE -ny $XY_SIZE -nz $z_size -iter $iter -streams $streams >> $OUTPUT_FILE 2>> $ERROR_FILE
            done
        done
    done
done

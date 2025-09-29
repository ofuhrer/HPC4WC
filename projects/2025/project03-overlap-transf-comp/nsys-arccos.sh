#!/bin/bash
#SBATCH --job-name=stencil_stream_profile
#SBATCH --output=measurements/stream_profile_%j.out
#SBATCH --error=measurements/stream_profile_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=debug

# Set up CUDA environment
export CUDA_ROOT=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3
export PATH=${CUDA_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:$LD_LIBRARY_PATH

# Add Nsight Systems to PATH
export NSYS_ROOT=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/profilers/Nsight_Systems
export PATH=${NSYS_ROOT}/bin:$PATH

cd build
mkdir -p ../measurements/profiles

# Check current working directory and create absolute path
PROFILE_DIR=$(realpath ../measurements/profiles)
echo "Profile directory: $PROFILE_DIR"
echo "Current working directory: $(pwd)"

# Profile different stream counts to see distribution
for streams in 4 8 16 32; do
    echo "Profiling with $streams streams..."
    OUTPUT_FILE="${PROFILE_DIR}/streams_${streams}.qdrep"
    echo "Output file will be: $OUTPUT_FILE"
    
    nsys profile \
        --output="$OUTPUT_FILE" \
        --export=sqlite \
        --force-overwrite=true \
        --trace=cuda,nvtx \
        --cuda-memory-usage=true \
        ./cuda_arccos 20 4096 $streams 1
    
    # Check if file was created
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Successfully created: $OUTPUT_FILE ($(ls -lh "$OUTPUT_FILE" | awk '{print $5}'))"
    else
        echo "ERROR: File not created at expected location"
        echo "Looking for any .qdrep files in current directory..."
        find . -name "*.qdrep" -type f -exec ls -lh {} \;
    fi
    echo "---"
done

echo "Final check of profile directory:"
ls -la "$PROFILE_DIR"

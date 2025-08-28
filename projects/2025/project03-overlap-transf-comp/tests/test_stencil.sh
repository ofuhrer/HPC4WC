#!/bin/bash
#SBATCH --job-name=stencil_stream_test
#SBATCH --output=../measurements/stream_test_%j.out
#SBATCH --error=../measurements/stream_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --partition=debug

# Set up CUDA environment
export CUDA_ROOT=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3
export PATH=${CUDA_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:$LD_LIBRARY_PATH

# run the stencil2d-gpu_streams.
mkdir -p ../measurements
mkdir -p build_tests
cd build_tests
/usr/bin/cmake -DCUDAToolkit_ROOT=${CUDA_ROOT} ../..

make clean
make stencil2d_gpu_streams
make stencil2d_cpu

# Test parameters (fixed variable names)
NX=128
NY=128
NZ=64
ITER=512

echo "Running baseline CPU version..."
./stencil2d_cpu -nx $NX -ny $NY -nz $NZ -iter $ITER
if [ $? -ne 0 ]; then
    echo "ERROR: CPU baseline failed!"
    exit 1
fi

echo "Running GPU streams version with 1 stream..."
./stencil2d_gpu_streams -nx $NX -ny $NY -nz $NZ -iter $ITER -streams 1 -test true
if [ $? -ne 0 ]; then
    echo "ERROR: GPU 1 stream failed!"
    exit 1
fi

echo "Comparing 1 stream output with CPU baseline..."
comparison_output=$(python3 ../comparison.py --src out_field.dat --trg out_field_streams.dat)
echo "$comparison_output"
if [[ "$comparison_output" != *"HOORAY"* ]]; then
    echo "ERROR: 1 stream comparison failed!"
    exit 1
fi

echo "Running GPU streams version with 2 streams..."
./stencil2d_gpu_streams -nx $NX -ny $NY -nz $NZ -iter $ITER -streams 2 -test true
if [ $? -ne 0 ]; then
    echo "ERROR: GPU 2 streams failed!"
    exit 1
fi

echo "Comparing 2 streams output with CPU baseline..."
comparison_output=$(python3 ../comparison.py --src out_field.dat --trg out_field_streams.dat)
echo "$comparison_output"
if [[ "$comparison_output" != *"HOORAY"* ]]; then
    echo "ERROR: 2 streams comparison failed!"
    exit 1
fi

echo "Running GPU streams version with 4 streams..."
./stencil2d_gpu_streams -nx $NX -ny $NY -nz $NZ -iter $ITER -streams 4 -test true
if [ $? -ne 0 ]; then
    echo "ERROR: GPU 4 streams failed!"
    exit 1
fi

echo "Comparing 4 streams output with CPU baseline..."
comparison_output=$(python3 ../comparison.py --src out_field.dat --trg out_field_streams.dat)
echo "$comparison_output"
if [[ "$comparison_output" != *"HOORAY"* ]]; then
    echo "ERROR: 4 streams comparison failed!"
    exit 1
fi

echo "All tests completed successfully!"
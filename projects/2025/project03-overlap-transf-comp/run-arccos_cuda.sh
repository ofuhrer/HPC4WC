#!/bin/bash
#SBATCH --job-name=perf_test_cuda
#SBATCH --output=measurements/perf_test_cuda_%j.out   # Save STDOUT to measurements dir
#SBATCH --error=measurements/perf_test_cuda_%j.err    # Save STDERR to measurements dir
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64              # Max CPU cores per task (adjust based on node)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=24:00:00                 # Max runtime
#SBATCH --exclusive                     # Get the whole node

echo "Job started on $(hostname)"
echo "Running performance tests for cuda"

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Set up CUDA environment
export CUDA_ROOT=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3
export PATH=${CUDA_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:$LD_LIBRARY_PATH

# Use system cmake and set CUDA toolkit root
/usr/bin/cmake -DCUDAToolkit_ROOT=${CUDA_ROOT} ..
make clean
make


# Create a timestamp for output file if not running via SLURM
if [ -z "$SLURM_JOB_ID" ]; then
    JOB_ID=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="../measurements/perf_test_cuda_${JOB_ID}.out"
    ERROR_FILE="../measurements/perf_test_cuda_${JOB_ID}.err"
    WARMUPOUT_FILE="../measurements/warmup_cuda_${JOB_ID}.out"
    WARMUPERR_FILE="../measurements/warmup_cuda_${JOB_ID}.err"
    # Ensure measurements directory exists
    mkdir -p ../measurements
else
    # When running via srun or sbatch, use absolute path from project root
    OUTPUT_FILE="../measurements/perf_test_cuda_$SLURM_JOB_ID.out"
    ERROR_FILE="../measurements/perf_test_cuda_$SLURM_JOB_ID.err"
    WARMUPOUT_FILE="../measurements/warmup_cuda_$SLURM_JOB_ID.out"
    WARMUPERR_FILE="../measurements/warmup_cuda_$SLURM_JOB_ID.err"
    # Ensure measurements directory exists
    mkdir -p ../measurements
fi

# warm up run with verification
./cuda_arccos 1 1024 4 10 1 >> $WARMUPOUT_FILE 2>> $WARMUPERR_FILE

# Check for errors in the warmup run
if grep -q "There were errors in the results or there was a runtime error." "$WARMUPERR_FILE"; then
    echo "Warm-up run failed. See $WARMUPERR_FILE for details."
    exit 1
else
    echo "Warm-up run completed successfully."
fi

echo "Saving results to: $OUTPUT_FILE"
echo "Beginning performance tests..." | tee -a $OUTPUT_FILE
echo "Errors will be logged to: $ERROR_FILE"

for ((rep=0; rep<10; rep+=1))
do
    REPS=$((2**rep))
    for ((k=3; k<30; k+= 2))
        do
        for ((i=0; i<10 && i<k; i+= 1))
            do
                SIZE=$((2**k))

                ./cuda_arccos $REPS $SIZE $((2**i)) 10 0 >> $OUTPUT_FILE 2>> $ERROR_FILE
                
            done
        done
done

source ~/HPC4WC_venv/bin/activate
python ../src/cuda/runtime_analysis.py $OUTPUT_FILE


echo "Job completed"
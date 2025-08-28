#!/bin/bash
#SBATCH --job-name=timeit_gt4py_DEBUG
#SBATCH --output=build/timeit_gt4py_DEBUG_%j.out   # Save STDOUT to build dir
#SBATCH --error=build/timeit_gt4py_DEBUG_%j.err    # Save STDERR to build dir
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64              # Max CPU cores per task (adjust based on node)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=00:06:00                 # Max runtime
#SBATCH --exclusive                     # Get the whole node


echo "Job started on $(hostname) -> Running performance tests for gt4py"

# Create directories if they don't exist
mkdir -p build
mkdir -p measurements

FILENAME="timeit_gt4py_DEBUG_$SLURM_JOB_ID"

OUTPUT_FILE="build/${FILENAME}.out"
ERROR_FILE="build/${FILENAME}.err"
CSV_FILE="measurements/${FILENAME}.csv"

# Choose backend via environment variable
export USE_BACKEND="GPU"

# # Choose size of debug run
# export DEBUG=1
export DEBUG="M"
# export DEBUG="L"

# # Uncomment if data transfer should be excluded from timing
# export NOTRANSFER=""

# Activate venv
source ~/HPC4WC_venv/bin/activate

# Check python path and version
echo "Using Python at: $(which python)"
python --version

# Run timing benchmarks
python build/gt4py/run_arccos_gt4py.py "$CSV_FILE" >> "$OUTPUT_FILE" 2>> "$ERROR_FILE"

if [[ $? -eq 1 ]]; then
    echo ""
    echo ""
    echo "** Python exited with error. See $ERROR_FILE. **"
    echo ""
fi

if [ -n "$SLURM_JOB_ID" ]; then
    sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,State
    echo ""
fi

echo "Job completed"

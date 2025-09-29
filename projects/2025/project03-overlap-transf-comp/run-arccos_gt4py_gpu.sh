#!/bin/bash
#SBATCH --job-name=timeit_gt4py_gpu
#SBATCH --output=build/timeit_gt4py_gpu_%j.out   # Save STDOUT to build dir
#SBATCH --error=build/timeit_gt4py_gpu_%j.err    # Save STDERR to build dir
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72              # Max CPU cores per task (adjust based on node)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=00:40:00                 # Max runtime
#SBATCH --exclusive                     # Get the whole node

echo "Job started on $(hostname)"
echo "Running performance tests for gt4py"

# Create directories if they don't exist
mkdir -p build
mkdir -p measurements

VERSION="gt4py_gpu"

# Create a timestamp for output file if not running via SLURM
if [ -z "$SLURM_JOB_ID" ]; then
    JOB_ID=$(date +%Y%m%d_%H%M%S)
    FILENAME="timeit_${VERSION}_${JOB_ID}"
    echo "Filenames for out and error files: $FILENAME (in build folder)"
else
    FILENAME="timeit_${VERSION}_$SLURM_JOB_ID"
fi

OUTPUT_FILE="build/${FILENAME}.out"
ERROR_FILE="build/${FILENAME}.err"
CSV_FILE="measurements/${FILENAME}.csv"

# Choose backend via environment variable
export USE_BACKEND="GPU"

# Activate venv
source ~/HPC4WC_venv/bin/activate

# Run timing benchmarks
python build/gt4py/run_arccos_gt4py.py "$CSV_FILE" >> "$OUTPUT_FILE" 2>> "$ERROR_FILE"

if [[ $? -eq 1 ]]; then
    echo ""
    echo ""
    echo "** Python exited with error. See $ERROR_FILE. **"
    echo ""
fi

echo ""
if [ -n "$SLURM_JOB_ID" ]; then
    sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,State
    echo ""
fi

echo "Job completed"
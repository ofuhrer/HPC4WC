#!/bin/bash
#SBATCH --job-name=stencil_gt4py_cpu
#SBATCH --output=build/stencil_gt4py_cpu_%j.out   # Save STDOUT to build dir
#SBATCH --error=build/stencil_gt4py_cpu_%j.err    # Save STDERR to build dir
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72              # Max CPU cores per task (adjust based on node)
#SBATCH --time=01:30:00                 # Max runtime
#SBATCH --exclusive                     # Get the whole node


echo "Job started on $(hostname) -> Running performance tests for gt4py"

# Create directories if they don't exist
mkdir -p build
mkdir -p measurements

FILENAME="stencil_gt4py_cpu_$SLURM_JOB_ID"

OUTPUT_FILE="build/${FILENAME}.out"
ERROR_FILE="build/${FILENAME}.err"
CSV_FILE="measurements/${FILENAME}.csv"

# Choose backend via environment variable
export USE_BACKEND="CPU"

# Activate venv
source ~/HPC4WC_venv/bin/activate

# Check Python path and version
echo "Using Python at: $(which python)"
python --version

# Run timing benchmarks
python build/gt4py/run_stencil2d_gt4py.py "$CSV_FILE" >> "$OUTPUT_FILE" 2>> "$ERROR_FILE"

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

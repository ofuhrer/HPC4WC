#!/bin/bash
#SBATCH --job-name=test_arccos_gt4py
#SBATCH --output=tests/test_arccos_gt4py_%j.out   # Save STDOUT to build dir
#SBATCH --error=tests/test_arccos_gt4py_%j.err    # Save STDERR to build dir
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64              # Max CPU cores per task (adjust based on node)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=00:40:00                 # Max runtime
#SBATCH --exclusive                     # Get the whole node


echo "Job started on $(hostname) -> Running accuracy tests for gt4py"

source ~/HPC4WC_venv/bin/activate

echo "Using Python at: $(which python)"
python --version

python build/gt4py/test_arccos_gt4py.py

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

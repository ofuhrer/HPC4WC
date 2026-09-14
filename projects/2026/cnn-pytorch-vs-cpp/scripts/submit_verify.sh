#!/bin/bash
#SBATCH --job-name=cnn_verify
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j_cnn_verify.out
#SBATCH --error=logs/%j_cnn_verify.err

cd $SLURM_SUBMIT_DIR
mkdir -p logs

# Explicitly set the number of threads to 1
export OMP_NUM_THREADS=1

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Start:   $(date)"

VENV="$HOME/venvs/venv"

make build
cd src/cpp && ../../build/cnn_forward verify && cd $SLURM_SUBMIT_DIR
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "source $VENV/bin/activate && cd src/python && python3 verify.py"

echo "End:     $(date)"

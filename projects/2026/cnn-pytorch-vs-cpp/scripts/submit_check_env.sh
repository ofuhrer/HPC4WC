#!/bin/bash
#SBATCH --job-name=check_env
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j_check_env.out
#SBATCH --error=logs/%j_check_env.err

mkdir -p $SLURM_SUBMIT_DIR/logs
cd $SLURM_SUBMIT_DIR/src/python

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

VENV="$HOME/venvs/venv"

uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "source $VENV/bin/activate && python3 -u check_env.py"

echo "End: $(date)"
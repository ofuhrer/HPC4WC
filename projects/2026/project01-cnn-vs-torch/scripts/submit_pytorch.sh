#!/bin/bash
#SBATCH --job-name=mnist_gpu
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j_mnist_gpu.out
#SBATCH --error=logs/%j_mnist_gpu.err

mkdir -p $SLURM_SUBMIT_DIR/logs
cd $SLURM_SUBMIT_DIR/src/python

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Device: GPU"
echo "Start: $(date)"


VENV="$HOME/venvs/venv"

uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "source $VENV/bin/activate && python3 -u train.py --device cuda"

echo "End: $(date)"
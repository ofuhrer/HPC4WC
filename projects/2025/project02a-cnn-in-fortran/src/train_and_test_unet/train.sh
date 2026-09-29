#!/bin/bash
#SBATCH --time=1-00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=train_Unet
#SBATCH --output=train_Unet-%J.log
#SBATCH --error=train_Unet.%j.err

export PYTHONUNBUFFERED=TRUE
export PATH="/users/class172/miniconda3/bin:$PATH"

source activate weather-cnn
conda run -n weather-cnn python -u train_and_test.py
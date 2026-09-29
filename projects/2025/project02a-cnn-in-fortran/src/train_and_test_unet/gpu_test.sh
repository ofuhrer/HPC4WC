#!/bin/bash
#SBATCH --time=1-00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=train_Unet_gpu_tst
#SBATCH --output=train_Unet-%J.log
#SBATCH --error=train_Unet.%j.err

module av
module spider
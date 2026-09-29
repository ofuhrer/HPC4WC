#!/bin/bash

# Load required modules
module load stack/.2024-04-silent
module load gcc/8.5.0
module load cuda/12.1.1
module load cudnn/8.9.7.29-12
module load libtorch/2.1.0
module load cmake/3.27.7

module list

echo "All necessary modules loaded!"
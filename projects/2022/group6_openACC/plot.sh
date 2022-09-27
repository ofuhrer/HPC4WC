#!bin/bash

# load modules for python plotting scripts
module load daint-gpu
module load matplotlib

# run python plotting scripts
python validation.py
python strong_scaling.py
python weak_scaling.py




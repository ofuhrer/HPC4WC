#!/bin/bash

num_iter=1024

# generate reference data
echo "running stencil2d-agnostic.py ..."
python stencil2d-agnostic.py --nx=512 --ny=512 --nz=64 --num_iter=${num_iter}

mv out_field.npy out_field_agnostic.npy

# run the programm to validate
echo "running stencil2d-parallel.py ..."
srun -n $SLURM_NNODES python stencil2d-parallel.py --nx=512 --ny=512 --nz=64 --num_iter=${num_iter} --plot_result true

# compare output againts control data
echo "running compare_fields.py ..."
python compare_fields.py --src="out_field_agnostic.npy" --trg="out_field.npy" --atol=0.6

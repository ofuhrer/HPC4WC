#!/bin/bash

num_iter=1024

# generate reference data
echo "running stencil2d.py ..."
python stencil2d-original.py --nx=128 --ny=128 --nz=64 --num_iter=${num_iter}

mv out_field.npy out_field_orig.npy

# run the programm to validate
echo "running stencil2d-agnostic.py ..."
CRAY_CUDA_MPS=1 python stencil2d-agnostic.py --nx=128 --ny=128 --nz=64 --num_iter=${num_iter} --plot_result=true

# compare output againts control data
echo "running compare_fields.py ..."
python compare_fields.py --src="out_field_orig.npy" --trg="out_field.npy"

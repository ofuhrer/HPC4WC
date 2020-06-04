#!/bin/bash

# generate reference data
echo "running stencil2d.py ..."
cd ../day1 && \
  python stencil2d.py --nx=32 --ny=32 --nz=64 --num_iter=16 && \
  cd ../day5 || exit

# run the programm to validate
echo "running stencil2d-gt4py.py ..."
python stencil2d-gt4py.py --nx=32 --ny=32 --nz=64 --num_iter=16

# compare output againts control data
echo "running compare_fields.py ..."
python compare_fields.py --src="../day1/out_field.npy" --trg="out_field.npy"

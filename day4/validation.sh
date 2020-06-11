#!/bin/bash

num_iter=1024

# generate reference data
echo "running stencil2d.py ..."
cd ../day1 && \
  python stencil2d.py --nx=128 --ny=128 --nz=64 --num_iter=${num_iter} && \
  cd ../day4 || exit

# run the programm to validate
echo "running stencil2d-cupy.py ..."
python stencil2d-cupy.py --nx=128 --ny=128 --nz=64 --num_iter=${num_iter}

# compare output againts control data
echo "running compare_fields.py ..."
python compare_fields.py --src="../day1/out_field.npy" --trg="out_field.npy"

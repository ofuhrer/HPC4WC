#!/bin/bash

VERSION=$1
BACKEND=$2

# generate reference data
echo "Running stencil2d.py ..."
cd ../day1 && \
  python stencil2d.py --nx=32 --ny=32 --nz=64 --num_iter=32 && \
  cd ../day5 || exit

# run the programm to validate
echo "Running stencil2d-gt4py-$VERSION.py ..."
python stencil2d-gt4py-$VERSION.py --nx=32 --ny=32 --nz=64 --num_iter=32 --backend=$BACKEND

# compare output againts control data
echo "Running compare_fields.py ..."
python compare_fields.py --src="../day1/out_field.npy" --trg="out_field.npy"

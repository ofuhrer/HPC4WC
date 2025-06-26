#!/bin/bash

if [ $# -ne 2 ] ; then
  echo "Usage: validation.sh <version> <backend>"
  exit 1
fi

VERSION=$1
BACKEND=$2
PWD=$(pwd)

# generate reference data
echo "Running stencil2d.py ..."
python stencil2d.py --nx=32 --ny=32 --nz=64 --num_iter=32 || return
/bin/mv in_field.npy in_field_ref.npy
/bin/mv out_field.npy out_field_ref.npy

# run the programm to validate
echo "Running stencil2d-gt4py-$VERSION.py ..."
rm -rf in_field.npy out_field.npy
python stencil2d-gt4py-v$VERSION.py --nx=32 --ny=32 --nz=64 --num_iter=32 --backend=$BACKEND || return

# compare output againts control data
echo "Running compare_fields.py ..."
python compare_fields.py --src="out_field_ref.npy" --trg="out_field.npy"

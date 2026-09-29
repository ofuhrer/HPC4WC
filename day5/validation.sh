#!/bin/bash

if [ $# -ne 2 ] ; then
  echo "Usage: validation.sh <version> <backend>"
  exit 1
fi

VERSION=${1#v}
BACKEND=$2

# generate reference data
echo "Running stencil2d.py ..."
python stencil2d.py --nx=32 --ny=32 --nz=64 --num_iter=32 || exit 1
/bin/mv in_field.npy in_field_ref.npy || exit 1
/bin/mv out_field.npy out_field_ref.npy || exit 1

# run the program to validate
echo "Running stencil2d-gt4py-v$VERSION.py ..."
rm -rf in_field.npy out_field.npy
python stencil2d-gt4py-v$VERSION.py --nx=32 --ny=32 --nz=64 --num_iter=32 --backend=$BACKEND || exit 1

# compare output against control data
echo "Running compare_fields.py ..."
python compare_fields.py --src="out_field_ref.npy" --trg="out_field.npy" || exit 1

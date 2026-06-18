#!/bin/bash

num_iter=1024

# generate reference data
echo "running stencil2d-agnostic.py ..."
python stencil2d-agnostic.py --nx=512 --ny=512 --nz=64 --num_iter=${num_iter}

mv out_field.npy out_field_agnostic.npy

# run the program to validate
echo "running stencil2d-parallel.py ..."
num_nodes=${SLURM_NNODES:-1}
gpus_per_node=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [ "${gpus_per_node}" -lt 1 ]; then
  gpus_per_node=1
fi
num_ranks=$((num_nodes * gpus_per_node))
srun -n ${num_ranks} -c 1 python stencil2d-parallel.py --nx=512 --ny=512 --nz=64 --num_iter=${num_iter} --plot_result true

# compare output against control data
echo "running compare_fields.py ..."
python compare_fields.py --src="out_field_agnostic.npy" --trg="out_field.npy" --atol=0.6

# hpc4wc:student-begin
# hpc4wc:student | #!/bin/bash
# hpc4wc:student |
# hpc4wc:student | num_iter=1024
# hpc4wc:student |
# hpc4wc:student | # generate reference data
# hpc4wc:student | echo "running stencil2d-agnostic.py ..."
# hpc4wc:student | python stencil2d-agnostic.py --nx=512 --ny=512 --nz=64 --num_iter=${num_iter}
# hpc4wc:student |
# hpc4wc:student | mv out_field.npy out_field_agnostic.npy
# hpc4wc:student |
# hpc4wc:student | # run the program to validate
# hpc4wc:student | echo "running stencil2d-parallel.py ..."
# hpc4wc:student | num_nodes=${SLURM_NNODES:-1}
# hpc4wc:student | gpus_per_node=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
# hpc4wc:student | if [ "${gpus_per_node}" -lt 1 ]; then
# hpc4wc:student |   gpus_per_node=1
# hpc4wc:student | fi
# hpc4wc:student | num_ranks=$((num_nodes * gpus_per_node))
# hpc4wc:student | srun -n ${num_ranks} -c 1 python stencil2d-parallel.py --nx=512 --ny=512 --nz=64 --num_iter=${num_iter} --plot_result true
# hpc4wc:student |
# hpc4wc:student | # compare output against control data
# hpc4wc:student | echo "running compare_fields.py ..."
# hpc4wc:student | python compare_fields.py --src="out_field_agnostic.npy" --trg="out_field.npy" --atol=0.6
# hpc4wc:student-end
# hpc4wc:solution-begin
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
# hpc4wc:solution-end

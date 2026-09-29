# hpc4wc:student-begin
# hpc4wc:student | #!/bin/bash
# hpc4wc:student |
# hpc4wc:student | num_iter=1024
# hpc4wc:student |
# hpc4wc:student | # generate reference data
# hpc4wc:student | echo "running stencil2d.py ..."
# hpc4wc:student | cd ../day1 && \
# hpc4wc:student |   python stencil2d.py --nx=128 --ny=128 --nz=64 --num_iter=${num_iter} && \
# hpc4wc:student |   cd ../day4 || exit
# hpc4wc:student |
# hpc4wc:student | if ! grep -q "^[[:space:]]*import cupy" stencil2d-cupy.py; then
# hpc4wc:student |   echo "stencil2d-cupy.py still does not import CuPy; finish Exercise 10 before validating."
# hpc4wc:student |   exit 1
# hpc4wc:student | fi
# hpc4wc:student |
# hpc4wc:student | # run the program to validate
# hpc4wc:student | echo "running stencil2d-cupy.py ..."
# hpc4wc:student | python stencil2d-cupy.py --nx=128 --ny=128 --nz=64 --num_iter=${num_iter}
# hpc4wc:student |
# hpc4wc:student | # compare output against control data
# hpc4wc:student | echo "running compare_fields.py ..."
# hpc4wc:student | python compare_fields.py --src="../day1/out_field.npy" --trg="out_field.npy"
# hpc4wc:student-end
# hpc4wc:solution-begin
#!/bin/bash

num_iter=1024

# generate reference data
echo "running stencil2d.py ..."
python stencil2d-original.py --nx=128 --ny=128 --nz=64 --num_iter=${num_iter}

mv out_field.npy out_field_orig.npy

# run the program to validate
echo "running stencil2d-cupy.py ..."
CRAY_CUDA_MPS=1 python stencil2d-cupy.py --nx=128 --ny=128 --nz=64 --num_iter=${num_iter}

# compare output against control data
echo "running compare_fields.py ..."
python compare_fields.py --src="out_field_orig.npy" --trg="out_field.npy"
# hpc4wc:solution-end

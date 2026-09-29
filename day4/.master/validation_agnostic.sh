# hpc4wc:student-begin
# hpc4wc:student | #!/bin/bash
# hpc4wc:student |
# hpc4wc:student | num_iter=1024
# hpc4wc:student |
# hpc4wc:student | # generate reference data
# hpc4wc:student | echo "running stencil2d.py ..."
# hpc4wc:student | python stencil2d-original.py --nx=128 --ny=128 --nz=64 --num_iter=${num_iter}
# hpc4wc:student |
# hpc4wc:student | mv out_field.npy out_field_orig.npy
# hpc4wc:student |
# hpc4wc:student | if ! grep -q "^[[:space:]]*import cupy" stencil2d-agnostic.py; then
# hpc4wc:student |   echo "stencil2d-agnostic.py still does not import CuPy; finish Bonus 12 before validating."
# hpc4wc:student |   exit 1
# hpc4wc:student | fi
# hpc4wc:student |
# hpc4wc:student | # run the program to validate
# hpc4wc:student | echo "running stencil2d-agnostic.py ..."
# hpc4wc:student | CRAY_CUDA_MPS=1 python stencil2d-agnostic.py --nx=128 --ny=128 --nz=64 --num_iter=${num_iter} --plot_result=true
# hpc4wc:student |
# hpc4wc:student | # compare output against control data
# hpc4wc:student | echo "running compare_fields.py ..."
# hpc4wc:student | python compare_fields.py --src="out_field_orig.npy" --trg="out_field.npy"
# hpc4wc:student-end
# hpc4wc:solution-begin
#!/bin/bash

num_iter=1024

# generate reference data
echo "running stencil2d.py ..."
python stencil2d-original.py --nx=128 --ny=128 --nz=64 --num_iter=${num_iter}

mv out_field.npy out_field_orig.npy

if ! grep -q "^[[:space:]]*import cupy" stencil2d-agnostic.py; then
  echo "stencil2d-agnostic.py still does not import CuPy; finish Bonus 12 before validating."
  exit 1
fi

# run the program to validate
echo "running stencil2d-agnostic.py ..."
CRAY_CUDA_MPS=1 python stencil2d-agnostic.py --nx=128 --ny=128 --nz=64 --num_iter=${num_iter} --plot_result=true

# compare output against control data
echo "running compare_fields.py ..."
python compare_fields.py --src="out_field_orig.npy" --trg="out_field.npy"
# hpc4wc:solution-end

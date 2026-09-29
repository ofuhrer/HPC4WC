#!/bin/bash
#
# GT4Py against Fortran, CPU and GPU backends.
#
# Authors: Stefanie Boersig <stefanie.boersig@env.ethz.ch>
#          Boaz Ko <boazko@student.ethz.ch>
#          Ben Bullinger <ben.bullinger@inf.ethz.ch>
#SBATCH --job-name=gt4py
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --time=00:28:00
#SBATCH --output=gt4py.out
# re-exec under the uenv so both the venv python and the Fortran binaries work
if [ -z "$UENV_WRAPPED" ]; then
  exec uenv run prgenv-gnu/26.3:v1 --view=default -- env UENV_WRAPPED=1 bash "$0"
fi
cd "${SLURM_SUBMIT_DIR:-$PWD}/src"
source "${HPC4WC_ENV:-$HOME/activate_hpc4wc.sh}"
export PYTHONOPTIMIZE=1
export OMP_NUM_THREADS=1
rm -f py_*.dat out_field.dat in_field.dat

echo "===== VALIDATION (64x64x16, 32 iters, vs Fortran singleton) ====="
for N in 2 4 6 8; do
  for LIM in "" "--limiter"; do
    ./stencil2d-filter.x --nx 64 --ny 64 --nz 16 --num_iter 32 --order $N $LIM > /dev/null 2>&1
    python3 gt4py/stencil2d-filter-gt4py.py --nx 64 --ny 64 --nz 16 --num_iter 32 --order $N $LIM --backend cpu > /dev/null 2>py_err.log
    printf "order=%s lim=%-9s " "$N" "${LIM:-off}"
    python3 ../tools/read_field.py out_field.dat py_out_field.dat --rtol 1e-5 --atol 1e-6
  done
done

echo "===== BENCHMARK (256x256x64, 128 iters, 1 core each) ====="
for N in 4 8; do
  for LIM in "" "--limiter"; do
    echo "### fortran-inline order=$N lim=${LIM:-off}"
    taskset -c 0 ./stencil2d-filter-inline.x --nx 256 --ny 256 --nz 64 --num_iter 128 --order $N $LIM --branchfree 2>/dev/null | grep "^\["
    echo "### gt4py-cpu order=$N lim=${LIM:-off}"
    taskset -c 0 python3 gt4py/stencil2d-filter-gt4py.py --nx 256 --ny 256 --nz 64 --num_iter 128 --order $N $LIM --backend cpu 2>>py_err.log | grep "^\["
    echo "### gt4py-gpu order=$N lim=${LIM:-off}"
    python3 gt4py/stencil2d-filter-gt4py.py --nx 256 --ny 256 --nz 64 --num_iter 128 --order $N $LIM --backend gpu 2>>py_err.log | grep "^\["
  done
done
echo ALLDONE

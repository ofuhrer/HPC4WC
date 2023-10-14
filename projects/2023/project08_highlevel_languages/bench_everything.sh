
# ================================================================================
# Requirements: 
# - Rust running, cargo installed 
# - Python installed
# - Julia installed
# 
# Note: The directory changes in this script happen because the paths in PythonCode
# and RustCode are relative to those folders respectively. This is convenient as it 
# means you can run those benchmarks in isolation from those folders. It also means
# we need to change directory to those folders if we run the scrips from this scrips
# in the directory hpcwc-performance-comparison.
# ================================================================================

echo "Setting python virtual environment"
source .venv/bin/activate

echo "==================== bench_everything: Python benchmarks"
cd PythonCode/src 
#python ./benches.py
#mpiexec -n 16 python -m mpi4py ./benches_mpi.py
cd ../../

echo "==================== bench_everything: Julia benchmarks"
#julia --threads 32 --project=JuliaCode -e "using JuliaCode; JuliaCode.run()"

echo "==================== bench_everything: Rust benchmarks"
cd RustCode/
set RAYON_NUM_THREADS=32
set RUSTFLAGS="-C target-cpu=native"
./bench_get_estimates.sh
cd ../../

echo "done"

source /users/class199/legate-shared/venv/bin/activate
module swap PrgEnv-cray PrgEnv-gnu/6.0.9
module swap gcc/10.1.0 gcc/9.3.0
module load cray-python/3.8.2.1
module load graphviz/2.44.0
module load daint-gpu
export NUMPY_BACKEND="LEGATE"
srun -A class03 -C gpu /users/class199/legate-shared/legate.core/install/bin/legate src/main.py --out data/legate_gpu.csv --libs numpy --ns  8 16 32 64 128 256 512 1024 2048 4096 8192 16384 --n_iter 4

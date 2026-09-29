#!/bin/bash
#SBATCH --job-name=cnn_run
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j_cnn_run.out
#SBATCH --error=logs/%j_cnn_run.err

cd $SLURM_SUBMIT_DIR
mkdir -p logs

# Explicitly set number of threads to 1
export OMP_NUM_THREADS=1

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Start:   $(date)"

# Defaults must match the Makefile's, for direct `sbatch scripts/submit_run.sh`.
NUM_RUNS="${NUM_RUNS:-30}"
NUM_WARMUP_RUNS="${NUM_WARMUP_RUNS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VENV="$HOME/venvs/venv"


uenv run --view=default pytorch/v2.9.1:v2 -- bash -c "
  echo '=== C++ Compiler Version ==='
  g++ --version | head -n1

  make build NUM_RUNS=$NUM_RUNS NUM_WARMUP_RUNS=$NUM_WARMUP_RUNS BATCH_SIZE=$BATCH_SIZE
  cd src/cpp && ../../build/cnn_forward
"


uenv run --view=default pytorch/v2.9.1:v2 -- bash -c "
  source $VENV/bin/activate
  cd src/python
  python3 benchmark.py --num-runs $NUM_RUNS --num-warmup-runs $NUM_WARMUP_RUNS --batch-size $BATCH_SIZE
"

echo "End:     $(date)"

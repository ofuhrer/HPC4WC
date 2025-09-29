#!/bin/bash

cd "$(dirname "$0")/.."

# Path to executable
GPU_BIN="./build-gpu/test_mnist_gpu"
mkdir -p logs

# Constants
BASE_SAMPLES=50000
MAX_CHUNK=5000 

# Output files
echo -e "GPUs\tTime(s)\tSpeedup\tEfficiency(%)" > logs/results_strong.tsv
echo -e "GPUs\tTime(s)\tSpeedup\tEfficiency(%)" > logs/results_weak.tsv

# Helper function
run_test() {
  local TOTAL=$1
  local CHUNK=$2
  local GPUS=$3
  local MODE=$4
  local LOG="logs/${MODE}_gpu_${GPUS}.out"

  START=$(date +%s.%N)
  export SLURM_NTASKS=$GPUS
  export SLURM_PROCID=0
  $GPU_BIN $TOTAL $CHUNK > "$LOG"
  END=$(date +%s.%N)
  TIME=$(echo "$END - $START" | bc)
  echo "$TIME"
}

### Strong Scaling
declare -A STRONG_TIMES
for GPUS in 1 2 4 8; do
  echo "Running strong scaling with $GPUS GPU(s)..."
  TOTAL=$BASE_SAMPLES
  CHUNK=$((BASE_SAMPLES / GPUS))
  if [ "$CHUNK" -gt "$MAX_CHUNK" ]; then
    CHUNK=$MAX_CHUNK
  fi
  TIME=$(run_test $TOTAL $CHUNK $GPUS "strong")
  STRONG_TIMES[$GPUS]=$TIME
done

# Compute speedup and efficiency (strong)
T1=${STRONG_TIMES[1]}
for GPUS in 1 2 4 8; do
  T=${STRONG_TIMES[$GPUS]}
  SPEEDUP=$(echo "$T1 / $T" | bc -l)
  EFFICIENCY=$(echo "100 * $SPEEDUP / $GPUS" | bc -l)
  printf "%d\t%.6f\t%.2f\t%.2f\n" "$GPUS" "$T" "$SPEEDUP" "$EFFICIENCY" >> logs/results_strong.tsv
done

### Weak Scaling
declare -A WEAK_TIMES
for GPUS in 1 2 4 8; do
  echo "Running weak scaling with $GPUS GPU(s)..."
  TOTAL=$((BASE_SAMPLES * GPUS))   # Increase total size with #GPUs
  CHUNK=$BASE_SAMPLES              # Keep work per GPU constant
  if [ "$CHUNK" -gt "$MAX_CHUNK" ]; then
    CHUNK=$MAX_CHUNK
  fi
  TIME=$(run_test $TOTAL $CHUNK $GPUS "weak")
  WEAK_TIMES[$GPUS]="$TIME:$TOTAL"
done

# Header with problem size
echo -e "GPUs\tTime(s)\tSpeedup\tEfficiency(%)\tTotal_Samples" > logs/results_weak.tsv

# Compute speedup and efficiency
IFS=':' read -r T1 T1_TOTAL <<< "${WEAK_TIMES[1]}"
for GPUS in 1 2 4 8; do
  IFS=':' read -r T TOTAL <<< "${WEAK_TIMES[$GPUS]}"
  SPEEDUP=$(echo "$T1 / $T" | bc -l)
  EFFICIENCY=$(echo "100 * $SPEEDUP / $GPUS" | bc -l)
  printf "%d\t%.6f\t%.2f\t%.2f\t%d\n" "$GPUS" "$T" "$SPEEDUP" "$EFFICIENCY" "$TOTAL" >> logs/results_weak.tsv
done


echo "GPU Scaling benchmarks completed. Results saved in logs/results_strong.tsv and logs/results_weak.tsv."

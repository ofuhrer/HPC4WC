#!/bin/bash
#SBATCH --job-name=1node_MPI_OMP_128_128     # Job name
#SBATCH --output=1node_MPI_OMP_128_128.out      # Output file (%j = job ID)
#SBATCH --error=1node_MPI_OMP_128_128.err       # Error file
#SBATCH --time=00:30:00            # Max run time (HH:MM:SS)
#SBATCH --partition=normal         # Partition/queue
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --exclusive

# Set OpenMP environment variables
export OMP_PLACES="cores"
export OMP_PROC_BIND="close"

nx_len=512
ny_len=512
echo "nx_len = $nx_len, ny_len = $ny_len"

file_name="perf"

out_file="experiments/out_${nx_len}_${ny_len}_${file_name}.txt"
dims_file="experiments/dims_${nx_len}_${ny_len}_${file_name}.txt"

if [[ $(hostname -s) == eiger-* ]]; then
  out_file="experiments/out_eiger_${nx_len}_${ny_len}_${file_name}.txt"
  dims_file="experiments/dims_eiger_${nx_len}_${ny_len}_${file_name}.txt"
fi

if [ ! -f "$out_file" ]; then
  echo "runtimes = [[[0.0]*73 for _ in range(73)]for _ in range(20)]" > "$out_file"
fi

if [ ! -f "$dims_file" ]; then
  echo "dimensions = [[[0.0]*73 for _ in range(73)]for _ in range(20)]" > "$dims_file"
fi
HINT="--hint=nomultithread"
BIND="--cpu-bind=cores"
for iteration in  1; do

    for nthreads in $(seq 72 72); do
        export OMP_NUM_THREADS=$nthreads
    
        for nnodes in $(seq 1 1); do
            if (( nthreads * nnodes > 72)); then
              continue
            fi
            echo "Running with $nthreads threads and $nnodes MPI tasks"
            if [[ $(hostname -s) == eiger-* ]]; then
                output=$(uenv run prgenv-gnu/24.11:v2 --view=default -- srun --ntasks=$nnodes --ntasks-per-node=$nnodes --cpus-per-task=$nthreads ./stencil2d-kparallel_mpi.x --nx 128 --ny 128 --nz 64 --num_iter 1024)
            else    
                # output=$(srun --ntasks=$nnodes --cpus-per-task=$nthreads ./stencil2d-kparallel_mpi.x --nx $nx_len --ny $ny_len --nz 64 --num_iter 1024)
                output=$(srun -N1 --ntasks=$nnodes --cpus-per-task=$nthreads ${HINT} ${BIND} --cpu-bind=verbose \
                 --bind-to core --map-by core  
                 ./perf_wrap.sh ./stencil2d-kparallel_mpi.x --nx $nx_len --ny $ny_len --nz 64 --num_iter 1024)
            fi    
            data_line=$(echo "$output" | grep -oP '\[\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*[\d.E+-]+\]')
            runtime=$(echo "$data_line" | sed 's/.*\[//' | sed 's/\]//' | cut -d',' -f6 | xargs)
            
            dims_line=$(echo "$output" | grep -oP '\[\s*\d+,\s*\d+\]' | head -n1)
            nx=$(echo "$dims_line" | sed 's/[][]//g' | cut -d',' -f1 | xargs)
            ny=$(echo "$dims_line" | sed 's/[][]//g' | cut -d',' -f2 | xargs)
            
            # Update or append runtime
            if grep -q "runtimes\[$iteration\]\[$nthreads\]\[$nnodes\]" "$out_file"; then
                sed -i "s/runtimes\[$iteration\]\[$nthreads\]\[$nnodes\] = .*/runtimes[$iteration][$nthreads][$nnodes] = $runtime/" "$out_file"
            else
                echo "runtimes[$iteration][$nthreads][$nnodes] = $runtime" >> "$out_file"
            fi
            
            # Update or append dimensions
            if grep -q "dimensions\[$nthreads\]\[$nnodes\]" "$dims_file"; then
                sed -i "s/dimensions\[$nthreads\]\[$nnodes\] = .*/dimensions[$nthreads][$nnodes] = [$nx, $ny]/" "$dims_file"
            else
                echo "dimensions[$nthreads][$nnodes] = [$nx, $ny]" >> "$dims_file"
            fi
        done
    done
done

#!/bin/bash -l
#SBATCH --job-name=baseline_2048_2048   # Job name
#SBATCH --output=error_outs/baseline_2048_2048.out      # Output file (%j = job ID)
#SBATCH --time=05:00:00            # Max run time (HH:MM:SS)
#SBATCH --partition=normal         # Partition/queue
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --exclusive

# Set OpenMP environment variables
export OMP_PLACES=cores
export OMP_PROC_BIND=close


nx_len=2048
ny_len=2048

nz_len=64


echo "nx_len = $nx_len, ny_len = $ny_len"

out_file="experiments_final/out_baseline_${nx_len}_${ny_len}.txt"
dims_file="experiments_final/dims_baseline_${nx_len}_${ny_len}.txt"

# if [ ! -f "$out_file" ]; then
echo "runtimes = [[[0.0]*73 for _ in range(73)]for _ in range(6)]" > "$out_file"
# fi

# if [ ! -f "$dims_file" ]; then
echo "dimensions = [[[0.0]*73 for _ in range(73)]for _ in range(6)]" > "$dims_file"
# fi

for iteration in $(seq 1 5); do
    for nthreads in $(seq 1 72); do
        for nranks in $(seq 1 72); do
        
            export OMP_NUM_THREADS=$nthreads
            if (( nthreads * nranks > 72)); then
              continue
            fi
            echo "Running with $nthreads threads and $nranks MPI tasks"

          # # prints per-rank CPU ranges & NUMA. (Sanity checks to make sure we are only running on one CPU, only ran once 
          # # and after checking I commented them out.)
          #   srun -N 1 -n $nranks --ntasks-per-node=$nranks --cpus-per-task=$nthreads --cpu-bind=verbose,cores \
          # --distribution=block:block --mem-bind=local -l bash -lc 'echo "rank=$SLURM_PROCID host=$(hostname)";
          #           echo -n "Cpus_allowed_list: "; awk "/Cpus_allowed_list/ {print \$2}" /proc/self/status;
          #           numactl --show | egrep "cpubind|membind"'
            
            output=$(srun -N 1 -n $nranks --ntasks-per-node=$nranks --cpus-per-task=$nthreads --cpu-bind=cores --distribution=block:block --mem-bind=local ./stencil2d-kparallel_mpi.x --nx $nx_len --ny $ny_len --nz $nz_len --num_iter 1024)

            data_line=$(echo "$output" | grep -oP '\[\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*[\d.E+-]+\]')
            runtime=$(echo "$data_line" | sed 's/.*\[//' | sed 's/\]//' | cut -d',' -f6 | xargs)
            
            dims_line=$(echo "$output" | grep -oP '\[\s*\d+,\s*\d+\]' | head -n1)
            nx=$(echo "$dims_line" | sed 's/[][]//g' | cut -d',' -f1 | xargs)
            ny=$(echo "$dims_line" | sed 's/[][]//g' | cut -d',' -f2 | xargs)
            
            # Update or append runtime
            if grep -q "runtimes\[$iteration\]\[$nthreads\]\[$nranks\]" "$out_file"; then
                sed -i "s/runtimes\[$iteration\]\[$nthreads\]\[$nranks\] = .*/runtimes[$iteration][$nthreads][$nranks] = $runtime/" "$out_file"
            else
                echo "runtimes[$iteration][$nthreads][$nranks] = $runtime" >> "$out_file"
            fi
            
            # Update or append dimensions
            if grep -q "dimensions\[$nthreads\]\[$nranks\]" "$dims_file"; then
                sed -i "s/dimensions\[$nthreads\]\[$nranks\] = .*/dimensions[$nthreads][$nranks] = [$nx, $ny]/" "$dims_file"
            else
                echo "dimensions[$nthreads][$nranks] = [$nx, $ny]" >> "$dims_file"
            fi
        done
    done
done
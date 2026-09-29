#!/bin/bash -l
#SBATCH --account=hpc4wc-course2026-ethz
#SBATCH --job-name="strong_scaling"
#SBATCH --output=out/strong_scaling.%j.o
#SBATCH --error=out/strong_scaling.%j.e
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --uenv=julia/25.5:v1     
#SBATCH --gpus-per-task=1

export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_CUDAAWARE_MPI=1
export JULIA_CUDA_USE_COMPAT=false

# Strong-scaling sizes with a 4:1 global aspect ratio.
# Nx/Ny values include the +2 halo padding (e.g. 2048+2 x 512+2 = 2050 x 514).

sizes=(
 
    "4098,1026"         # ny-2=1024, nx-2=4096  
    "6146,1538"         # ny-2=1536, nx-2=6144 
    "8194,2050"         # ny-2=2048, nx-2=8192  
    "10242,2562"        # ny-2=2560, nx-2=10240 
    "12802,3202"        # ny-2=3200, nx-2=12800

    "16386,4098"        # ny-2=4096, nx-2=16384 
    "20482,5122"        # ny-2=5120, nx-2=20480 
    "24578,6146"        # ny-2=6144, nx-2=24576
    "32770,8194"        # ny-2=8192, nx-2=32768 
    "40962,10242"       # ny-2=10240,nx-2=40960 

    "49154,12290"       # ny-2=12288,nx-2=49152
    "65538,16386"       # ny-2=16384,nx-2=65536 
    "81922,20482"       # ny-2=20480,nx-2=81920 
    "98306,24578"       # ny-2=24576,nx-2=98304 
    "131074,32770"      # ny-2=32768,nx-2=131072

    "163842,40962"      # ny-2=40960,nx-2=163840
    "196610,49154"      # ny-2=49152,nx-2=196608
    "229378,57346"      # ny-2=57344,nx-2=229376
)

topologies=(
    "(16,1)"
    "(8,2)"
    "(4,4)"
    "(2,8)"
    "(1,16)"
)

manual_bench="output/strong_scaling_manual.csv"
baseline_bench="output/benchmark/strong_scaling_baseline.csv"


mkdir -p "$(dirname "$manual_bench")" "$manual_outdir_root" "$baseline_outdir_root"

for size in "${sizes[@]}"; do
    IFS=',' read -r nx ny <<< "$size"

    # --- 1. MANUAL RUNS (Nutzt intern --runs 5, nur 1 Julia-Start pro Topologie!) ---
    for topo in "${topologies[@]}"; do
        topo_clean=$(echo "${topo}" | tr -d '(),')
        outdir="${manual_outdir_root}/nx_${nx}_ny_${ny}/topo_${topo_clean}"
        mkdir -p "${outdir}"

        srun --export=ALL,USE_GPU=true julia --project manual.jl \
            --topo "${topo}" \
            --nx "${nx}" \
            --ny "${ny}" \
            --nt 20 \
            --warmup 5 \
            --runs 5 \
            --benchmark \
            --benchdir "${manual_bench}" \
            --outdir "${outdir}"
    done

    # --- 2. BASELINE RUNS (Kein --runs Argument vorhanden, daher 5-mal in Bash gestartet) ---
    for run in {1..5}; do
        outdir_base="${baseline_outdir_root}/run_${run}/nx_${nx}_ny_${ny}"
        mkdir -p "${outdir_base}"

        srun --export=ALL,USE_GPU=true julia --project baseline.jl \
            --nx "${nx}" \
            --ny "${ny}" \
            --nt 20 \
            --warmup 5 \
            --benchmark \
            --benchdir "${baseline_bench}" \
            --outdir "${outdir_base}"
    done
done
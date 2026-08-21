# Benchmarks

    machine             one GH200 node of Alps (CSCS, vCluster Santis)
    CPU                 72 MPI ranks on one NVIDIA Grace (Neoverse-V2),
                        unless a note says otherwise
    toolchain           uenv prgenv-gnu/26.3:v1, gfortran 14.3,
                        -O3 -mcpu=native
    precision           single (wp = 4)
    raw .out logs       not kept; results.md carries the parsed medians

    layout              one directory per experiment
    run.sh              submits it
    counters.sh         collects hardware counters
    results.md          holds the numbers
    limiter/            predates the batch workflow: interactive srun,
                        same settings
    submit from         the project directory, so SLURM_SUBMIT_DIR resolves
                        sbatch benchmarks/orders/run.sh
    GT4Py job needs     the course Python environment; set HPC4WC_ENV if it
                        is not at ~/activate_hpc4wc.sh
    re-parse a log      python3 ../tools/parse_bench.py <job-output>.out

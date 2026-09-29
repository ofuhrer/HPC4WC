# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute avg_pack avg_send avg_wait
data = np.array( [ \
[   16,  512,  512,   64,    1024,  0.1397232E+01,  0.2408199E+00,  0.1219537E+01,  0.2145553E+00,  0.1182551E+01,  0.1164207E+00,  0.9984991E-02,  0.8785487E-01], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "corners"
nodes = 4
ranks = 16
cores_per_rank = 1
mode = "weak"
rep = 3
experiment = "results_weak_128_detailed_timing"
binary = "./stencil2d-mpi-corners.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 4 -n 16 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10 ./stencil2d-mpi-corners.x --nx 512 --ny 512 --nz 64 --num_iter 1024"
env_overrides = "OMP_NUM_THREADS=1"
srun_extra = "--ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10"
app_extra = ""
git_commit = "5f659f41dc0b2cbd347870eaa2405c47714603f4"
git_branch = "main"
git_dirty = 1
git_diff = "results_weak_128_detailed_timing/code_state.diff"
git_status = "results_weak_128_detailed_timing/code_state.status"
sweep_start = "2026-08-31T12:27:43+02:00"
sweep_command = "./experiment_scaling_all_strategies.sh --mode weak --ranks 16\\,32\\,64 --results-dir results_weak_128_detailed_timing"
slurm_jobid = "837127"
t_start = "2026-08-31T12:30:43+02:00"
wall_s = 28.410
t_start_approx = 0
metadata_source = "recorded"
local_nx = 128
local_ny = 128

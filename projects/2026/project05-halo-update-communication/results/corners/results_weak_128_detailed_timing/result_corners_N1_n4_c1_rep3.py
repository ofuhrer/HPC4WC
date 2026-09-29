# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute avg_pack avg_send avg_wait
data = np.array( [ \
[    4,  256,  256,   64,    1024,  0.1322974E+01,  0.2054698E+00,  0.1135168E+01,  0.1948934E+00,  0.1127960E+01,  0.1282169E+00,  0.1390908E-01,  0.5256505E-01], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "corners"
nodes = 1
ranks = 4
cores_per_rank = 1
mode = "weak"
rep = 3
experiment = "results_weak_128_detailed_timing"
binary = "./stencil2d-mpi-corners.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 1 -n 4 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10 ./stencil2d-mpi-corners.x --nx 256 --ny 256 --nz 64 --num_iter 1024"
env_overrides = "OMP_NUM_THREADS=1"
srun_extra = "--ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10"
app_extra = ""
git_commit = "06d245a2a1b44d63cdba3f3f9e8b2bf007f444e0"
git_branch = "main"
git_dirty = 1
git_diff = "results_weak_128_detailed_timing/code_state.diff"
git_status = "results_weak_128_detailed_timing/code_state.status"
sweep_start = "2026-08-31T11:39:37+02:00"
sweep_command = "./experiment_scaling_all_strategies.sh --mode weak --ranks 1\\,2\\,4\\,8 --results-dir results_weak_128_detailed_timing"
slurm_jobid = "836833"
t_start = "2026-08-31T11:51:35+02:00"
wall_s = 29.269
t_start_approx = 0
metadata_source = "recorded"
local_nx = 128
local_ny = 128

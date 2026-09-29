# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute avg_pack avg_send avg_wait
data = np.array( [ \
[    2,  128,  256,   64,    1024,  0.1280531E+01,  0.1531953E+00,  0.1130231E+01,  0.1516044E+00,  0.1128739E+01,  0.9687818E-01,  0.1162212E-01,  0.4155573E-01], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "waitany"
nodes = 1
ranks = 2
cores_per_rank = 1
mode = "weak"
rep = 2
experiment = "results_weak_128_detailed_timing"
binary = "./stencil2d-mpi-corners-waitany.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 1 -n 2 -c 1 --ntasks-per-node=2 --gpus-per-task=1 --cpu-bind=cores -t 10 ./stencil2d-mpi-corners-waitany.x --nx 128 --ny 256 --nz 64 --num_iter 1024"
env_overrides = "OMP_NUM_THREADS=1"
srun_extra = "--ntasks-per-node=2 --gpus-per-task=1 --cpu-bind=cores -t 10"
app_extra = ""
git_commit = "06d245a2a1b44d63cdba3f3f9e8b2bf007f444e0"
git_branch = "main"
git_dirty = 1
git_diff = "results_weak_128_detailed_timing/code_state.diff"
git_status = "results_weak_128_detailed_timing/code_state.status"
sweep_start = "2026-08-31T11:39:37+02:00"
sweep_command = "./experiment_scaling_all_strategies.sh --mode weak --ranks 1\\,2\\,4\\,8 --results-dir results_weak_128_detailed_timing"
slurm_jobid = "836818"
t_start = "2026-08-31T11:46:35+02:00"
wall_s = 29.745
t_start_approx = 0
metadata_source = "recorded"
local_nx = 128
local_ny = 128

# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute avg_pack avg_send avg_wait
data = np.array( [ \
[   96, 1024, 1536,   64,    1024,  0.1386065E+01,  0.2993883E+00,  0.1161335E+01,  0.2542401E+00,  0.1131700E+01,  0.1124854E+00,  0.9223996E-02,  0.1322283E+00], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "corners"
nodes = 24
ranks = 96
cores_per_rank = 1
mode = "weak"
rep = 3
experiment = "results_weak_128_detailed_timing"
binary = "./stencil2d-mpi-corners.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 24 -n 96 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10 ./stencil2d-mpi-corners.x --nx 1024 --ny 1536 --nz 64 --num_iter 1024"
env_overrides = "OMP_NUM_THREADS=1"
srun_extra = "--ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10"
app_extra = ""
git_commit = "5f659f41dc0b2cbd347870eaa2405c47714603f4"
git_branch = "main"
git_dirty = 1
git_diff = "results_weak_128_detailed_timing/code_state.diff"
git_status = "results_weak_128_detailed_timing/code_state.status"
sweep_start = "2026-08-31T14:04:55+02:00"
sweep_command = "./experiment_scaling_all_strategies.sh --mode weak --ranks 96 --local-n 128 --nz 64 --num_iter 1024 --repeats 3 --results-dir results_weak_128_detailed_timing"
slurm_jobid = "837463"
t_start = "2026-08-31T14:07:27+02:00"
wall_s = 26.064
t_start_approx = 0
metadata_source = "recorded"
local_nx = 128
local_ny = 128

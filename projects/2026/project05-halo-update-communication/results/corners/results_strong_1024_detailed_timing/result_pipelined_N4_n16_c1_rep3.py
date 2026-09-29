# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute avg_pack avg_send avg_wait
data = np.array( [ \
[   16, 1024, 1024,   64,    1024,  0.4435342E+01,  0.5826843E+00,  0.4138296E+01,  0.4613804E+00,  0.3973770E+01,  0.2100324E+00,  0.3031889E-01,  0.2206521E+00], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "pipelined"
nodes = 4
ranks = 16
cores_per_rank = 1
mode = "strong"
rep = 3
experiment = "results_strong_1024_detailed_timing"
binary = "./stencil2d-mpi-corners-pipelined.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 4 -n 16 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 15 ./stencil2d-mpi-corners-pipelined.x --nx 1024 --ny 1024 --nz 64 --num_iter 1024"
env_overrides = "OMP_NUM_THREADS=1"
srun_extra = "--ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 15"
app_extra = ""
git_commit = "5f659f41dc0b2cbd347870eaa2405c47714603f4"
git_branch = "main"
git_dirty = 1
git_diff = "results_strong_1024_detailed_timing/code_state.diff"
git_status = "results_strong_1024_detailed_timing/code_state.status"
sweep_start = "2026-08-31T12:44:25+02:00"
sweep_command = "./experiment_scaling_all_strategies.sh --mode strong --ranks 64\\,32\\,16\\,8\\,4\\,2\\,1 --nx 1024 --ny 1024 --nz 64 --num_iter 1024 --repeats 3 --results-dir results_strong_1024_detailed_timing"
slurm_jobid = "837261"
t_start = "2026-08-31T13:14:20+02:00"
wall_s = 22.759
t_start_approx = 0
metadata_source = "recorded"

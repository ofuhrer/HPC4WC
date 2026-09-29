# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute avg_pack avg_send avg_wait
data = np.array( [ \
[    2, 1024, 1024,   64,    1024,  0.4472230E+02,  0.1544815E+01,  0.4371648E+02,  0.1275026E+01,  0.4344667E+02,  0.7777948E+00,  0.9455483E-01,  0.4021786E+00], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "pipelined"
nodes = 1
ranks = 2
cores_per_rank = 1
mode = "strong"
rep = 3
experiment = "results_strong_1024_detailed_timing"
binary = "./stencil2d-mpi-corners-pipelined.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 1 -n 2 -c 1 --ntasks-per-node=2 --gpus-per-task=1 --cpu-bind=cores -t 15 ./stencil2d-mpi-corners-pipelined.x --nx 1024 --ny 1024 --nz 64 --num_iter 1024"
env_overrides = "OMP_NUM_THREADS=1"
srun_extra = "--ntasks-per-node=2 --gpus-per-task=1 --cpu-bind=cores -t 15"
app_extra = ""
git_commit = "5f659f41dc0b2cbd347870eaa2405c47714603f4"
git_branch = "main"
git_dirty = 1
git_diff = "results_strong_1024_detailed_timing/code_state.diff"
git_status = "results_strong_1024_detailed_timing/code_state.status"
sweep_start = "2026-08-31T12:44:25+02:00"
sweep_command = "./experiment_scaling_all_strategies.sh --mode strong --ranks 64\\,32\\,16\\,8\\,4\\,2\\,1 --nx 1024 --ny 1024 --nz 64 --num_iter 1024 --repeats 3 --results-dir results_strong_1024_detailed_timing"
slurm_jobid = "837295"
t_start = "2026-08-31T13:32:41+02:00"
wall_s = 62.207
t_start_approx = 0
metadata_source = "recorded"

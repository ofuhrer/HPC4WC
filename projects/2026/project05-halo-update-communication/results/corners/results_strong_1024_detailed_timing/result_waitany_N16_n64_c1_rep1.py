# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute avg_pack avg_send avg_wait
data = np.array( [ \
[   64, 1024, 1024,   64,    1024,  0.1340972E+01,  0.2462620E+00,  0.1161219E+01,  0.2056613E+00,  0.1135178E+01,  0.1238632E+00,  0.9840767E-02,  0.7046025E-01], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "waitany"
nodes = 16
ranks = 64
cores_per_rank = 1
mode = "strong"
rep = 1
experiment = "results_strong_1024_detailed_timing"
binary = "./stencil2d-mpi-corners-waitany.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 16 -n 64 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 15 ./stencil2d-mpi-corners-waitany.x --nx 1024 --ny 1024 --nz 64 --num_iter 1024"
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
slurm_jobid = "837160"
t_start = "2026-08-31T12:45:18+02:00"
wall_s = 32.912
t_start_approx = 0
metadata_source = "recorded"

# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute avg_pack avg_send avg_wait
data = np.array( [ \
[   96, 1024, 1024,   64,    1024,  0.9413578E+00,  0.2216708E+00,  0.7908009E+00,  0.1852982E+00,  0.7559269E+00,  0.9069105E-01,  0.9665342E-02,  0.8342367E-01], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "waitany"
nodes = 24
ranks = 96
cores_per_rank = 1
mode = "strong"
rep = 1
experiment = "results_corners_strong_1024_timed_n96"
binary = "./stencil2d-mpi-corners-waitany.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 24 -n 96 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores ./stencil2d-mpi-corners-waitany.x --nx 1024 --ny 1024 --nz 64 --num_iter 1024"
env_overrides = "OMP_NUM_THREADS=1"
srun_extra = "--ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores"
app_extra = ""
git_commit = "5f659f41dc0b2cbd347870eaa2405c47714603f4"
git_branch = "main"
git_dirty = 1
git_diff = "results_corners_strong_1024_timed_n96/code_state.diff"
git_status = "results_corners_strong_1024_timed_n96/code_state.status"
sweep_start = "2026-08-31T14:41:46+02:00"
sweep_command = "./experiment_corners_strong_all.sh --ranks 96 --repeats 3 --results-dir results_corners_strong_1024_timed_n96 --no-build"
slurm_jobid = "837623"
t_start = "2026-08-31T14:42:34+02:00"
wall_s = 26.745
t_start_approx = 0
metadata_source = "recorded"

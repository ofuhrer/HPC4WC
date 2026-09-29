# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute avg_pack avg_send avg_wait
data = np.array( [ \
[   96, 1024, 1536,   64,    1024,  0.1425488E+01,  0.3008406E+00,  0.1221704E+01,  0.2510112E+00,  0.1174345E+01,  0.1120444E+00,  0.6519397E-02,  0.1321092E+00], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "mpi_leftright"
nodes = 24
ranks = 96
cores_per_rank = 1
mode = "weak"
rep = 3
experiment = "results_weak_128_detailed_timing"
binary = "./stencil2d-mpi.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 24 -n 96 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10 ./stencil2d-mpi.x --nx 1024 --ny 1536 --nz 64 --num_iter 1024"
env_overrides = "OMP_NUM_THREADS=1"
srun_extra = "--ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10"
app_extra = ""
git_commit = "5f659f41dc0b2cbd347870eaa2405c47714603f4"
git_branch = "main"
git_dirty = 1
git_diff = "results_weak_128_detailed_timing/code_state.diff"
git_status = "results_weak_128_detailed_timing/code_state.status"
sweep_start = "2026-08-31T14:04:55+02:00"
sweep_command = "./experiment_mpi_leftright_scaling.sh --mode weak --ranks 96 --local-n 128 --nz 64 --num_iter 1024 --repeats 3 --results-dir results_weak_128_detailed_timing"
slurm_jobid = "837380"
t_start = "2026-08-31T14:05:42+02:00"
wall_s = 28.200
t_start_approx = 0
metadata_source = "recorded"
local_nx = 128
local_ny = 128

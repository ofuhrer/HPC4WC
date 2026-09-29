# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute avg_pack avg_send avg_wait
data = np.array( [ \
[   64, 1024, 1024,   64,    1024,  0.1341837E+01,  0.2480524E+00,  0.1161500E+01,  0.2093855E+00,  0.1132322E+01,  0.1143786E+00,  0.9472458E-02,  0.8521351E-01], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "corners"
nodes = 16
ranks = 64
cores_per_rank = 1
mode = "weak"
rep = 2
experiment = "results_weak_128_detailed_timing"
binary = "./stencil2d-mpi-corners.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 16 -n 64 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10 ./stencil2d-mpi-corners.x --nx 1024 --ny 1024 --nz 64 --num_iter 1024"
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
slurm_jobid = "837148"
t_start = "2026-08-31T12:38:18+02:00"
wall_s = 29.954
t_start_approx = 0
metadata_source = "recorded"
local_nx = 128
local_ny = 128

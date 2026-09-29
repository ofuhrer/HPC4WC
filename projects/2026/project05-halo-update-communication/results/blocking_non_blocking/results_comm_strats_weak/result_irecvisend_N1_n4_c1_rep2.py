# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = irecvisend
data = np.array( [ \
[    4,  256,  256,   64,    1024,  0.1267365E+01,  0.1700531E+00,  0.8753642E+00,  0.1390660E+00,  0.8680130E+00], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "irecvisend"
nodes = 1
ranks = 4
cores_per_rank = 1
mode = "weak"
rep = 2
experiment = "results_comm_strats_weak"
binary = "./stencil2d-comm_strats_timer.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 1 -n 4 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10 ./stencil2d-comm_strats_timer.x --nx 256 --ny 256 --nz 64 --num_iter 1024 --comm_strategy irecvisend"
env_overrides = "OMP_NUM_THREADS=1"
srun_extra = "--ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10"
app_extra = "--comm_strategy irecvisend"
git_commit = "85786f911054318eb65412916fee493cde4e3336"
git_branch = "main"
git_dirty = 1
git_diff = "results_comm_strats_weak/code_state.diff"
git_status = "results_comm_strats_weak/code_state.status"
sweep_start = "2026-09-15T14:46:40+02:00"
sweep_command = "./experiment_scaling_all_strategies.sh --mode weak --ranks 4\\,8\\,16\\,32\\,64 --local-n 128 --nz 64 --num_iter 1024 --repeats 3 --results-dir results_comm_strats_weak"
slurm_jobid = "866593"
t_start = "2026-09-15T14:49:58+02:00"
wall_s = 29.962
t_start_approx = 0
metadata_source = "recorded"
local_nx = 128
local_ny = 128

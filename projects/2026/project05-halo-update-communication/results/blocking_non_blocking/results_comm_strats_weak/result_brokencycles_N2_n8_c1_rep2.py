# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = brokencycles
data = np.array( [ \
[    8,  256,  512,   64,    1024,  0.1358035E+01,  0.2312186E+00,  0.8834711E+00,  0.2226866E+00,  0.8751267E+00], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "brokencycles"
nodes = 2
ranks = 8
cores_per_rank = 1
mode = "weak"
rep = 2
experiment = "results_comm_strats_weak"
binary = "./stencil2d-comm_strats_timer.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 2 -n 8 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10 ./stencil2d-comm_strats_timer.x --nx 256 --ny 512 --nz 64 --num_iter 1024 --comm_strategy brokencycles"
env_overrides = "OMP_NUM_THREADS=1"
srun_extra = "--ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10"
app_extra = "--comm_strategy brokencycles"
git_commit = "85786f911054318eb65412916fee493cde4e3336"
git_branch = "main"
git_dirty = 1
git_diff = "results_comm_strats_weak/code_state.diff"
git_status = "results_comm_strats_weak/code_state.status"
sweep_start = "2026-09-15T14:46:40+02:00"
sweep_command = "./experiment_scaling_all_strategies.sh --mode weak --ranks 4\\,8\\,16\\,32\\,64 --local-n 128 --nz 64 --num_iter 1024 --repeats 3 --results-dir results_comm_strats_weak"
slurm_jobid = "866606"
t_start = "2026-09-15T14:54:51+02:00"
wall_s = 19.477
t_start_approx = 0
metadata_source = "recorded"
local_nx = 128
local_ny = 128

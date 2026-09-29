# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = sendrecv_evenodd
data = np.array( [ \
[    8, 1024, 1024,   64,    1024,  0.1000595E+02,  0.6187659E+00,  0.7893402E+01,  0.5464609E+00,  0.7804208E+01], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "sendrecv_evenodd"
nodes = 2
ranks = 8
cores_per_rank = 1
mode = "strong"
rep = 1
experiment = "results_comm_strats_strong"
binary = "./stencil2d-comm_strats_timer.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 2 -n 8 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 15 ./stencil2d-comm_strats_timer.x --nx 1024 --ny 1024 --nz 64 --num_iter 1024 --comm_strategy sendrecv_evenodd"
env_overrides = "OMP_NUM_THREADS=1"
srun_extra = "--ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 15"
app_extra = "--comm_strategy sendrecv_evenodd"
git_commit = "85786f911054318eb65412916fee493cde4e3336"
git_branch = "main"
git_dirty = 1
git_diff = "results_comm_strats_strong/code_state.diff"
git_status = "results_comm_strats_strong/code_state.status"
sweep_start = "2026-09-15T15:52:53+02:00"
sweep_command = "./experiment_scaling_all_strategies.sh --mode strong --ranks 4\\,8\\,16\\,32\\,64 --nx 1024 --ny 1024 --nz 64 --num_iter 1024 --repeats 3 --results-dir results_comm_strats_strong"
slurm_jobid = "867048"
t_start = "2026-09-15T16:03:48+02:00"
wall_s = 31.102
t_start_approx = 0
metadata_source = "recorded"

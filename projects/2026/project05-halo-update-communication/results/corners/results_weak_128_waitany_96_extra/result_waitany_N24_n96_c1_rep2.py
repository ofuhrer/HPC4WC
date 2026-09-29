# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute avg_pack avg_send avg_wait
data = np.array( [ \
[   96, 1024, 1536,   64,    1024,  0.1351940E+01,  0.2590838E+00,  0.1158988E+01,  0.2169143E+00,  0.1134891E+01,  0.1241904E+00,  0.9777771E-02,  0.8144955E-01], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "waitany"
nodes = 24
ranks = 96
cores_per_rank = 1
mode = "weak"
rep = 2
experiment = "results_weak_128_waitany_96_extra"
binary = "./stencil2d-mpi-corners-waitany.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 24 -n 96 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10 ./stencil2d-mpi-corners-waitany.x --nx 1024 --ny 1536 --nz 64 --num_iter 1024"
env_overrides = "OMP_NUM_THREADS=1"
srun_extra = "--ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10"
app_extra = ""
git_commit = "5f659f41dc0b2cbd347870eaa2405c47714603f4"
git_branch = "main"
git_dirty = 1
git_diff = "results_weak_128_waitany_96_extra/code_state.diff"
git_status = "results_weak_128_waitany_96_extra/code_state.status"
sweep_start = "2026-08-31T15:22:02+02:00"
sweep_command = "./experiment_scaling_all_strategies.sh --mode weak --ranks 96 --local-n 128 --nz 64 --num_iter 1024 --repeats 3 --results-dir results_weak_128_waitany_96_extra"
slurm_jobid = "837749"
t_start = "2026-08-31T15:23:00+02:00"
wall_s = 31.310
t_start_approx = 0
metadata_source = "recorded"
local_nx = 128
local_ny = 128

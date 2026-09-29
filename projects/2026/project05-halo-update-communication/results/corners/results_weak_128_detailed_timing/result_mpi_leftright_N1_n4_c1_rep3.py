# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute avg_pack avg_send avg_wait
data = np.array( [ \
[    4,  256,  256,   64,    1024,  0.1332348E+01,  0.2160260E+00,  0.1144129E+01,  0.2027766E+00,  0.1129452E+01,  0.1367046E+00,  0.5724681E-02,  0.6005225E-01], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "mpi_leftright"
nodes = 1
ranks = 4
cores_per_rank = 1
mode = "weak"
rep = 3
experiment = "results_weak_128_detailed_timing"
binary = "./stencil2d-mpi.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 1 -n 4 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10 ./stencil2d-mpi.x --nx 256 --ny 256 --nz 64 --num_iter 1024"
env_overrides = "OMP_NUM_THREADS=1"
srun_extra = "--ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10"
app_extra = ""
git_commit = "06d245a2a1b44d63cdba3f3f9e8b2bf007f444e0"
git_branch = "main"
git_dirty = 1
git_diff = "results_weak_128_detailed_timing/code_state.diff"
git_status = "results_weak_128_detailed_timing/code_state.status"
sweep_start = "2026-08-31T11:33:42+02:00"
sweep_command = "./experiment_mpi_leftright_scaling.sh --mode weak --ranks 1\\,2\\,4\\,8 --results-dir results_weak_128_detailed_timing"
slurm_jobid = "836761"
t_start = "2026-08-31T11:37:34+02:00"
wall_s = 29.490
t_start_approx = 0
metadata_source = "recorded"
local_nx = 128
local_ny = 128

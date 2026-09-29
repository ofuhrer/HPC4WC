# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute avg_pack avg_send avg_wait
data = np.array( [ \
[    8,  256,  512,   64,    1024,  0.1322376E+01,  0.2033441E+00,  0.1143183E+01,  0.1907672E+00,  0.1131461E+01,  0.1280416E+00,  0.6971307E-02,  0.5538814E-01], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "mpi_leftright"
nodes = 2
ranks = 8
cores_per_rank = 1
mode = "weak"
rep = 1
experiment = "results_weak_128_detailed_timing"
binary = "./stencil2d-mpi.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 2 -n 8 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 10 ./stencil2d-mpi.x --nx 256 --ny 512 --nz 64 --num_iter 1024"
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
slurm_jobid = "836763"
t_start = "2026-08-31T11:38:04+02:00"
wall_s = 32.766
t_start_approx = 0
metadata_source = "recorded"
local_nx = 128
local_ny = 128

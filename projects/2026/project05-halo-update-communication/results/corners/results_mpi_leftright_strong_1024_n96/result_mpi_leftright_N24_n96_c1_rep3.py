# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute avg_pack avg_send avg_wait
data = np.array( [ \
[   96, 1024, 1024,   64,    1024,  0.9509472E+00,  0.2364871E+00,  0.7853338E+00,  0.1957047E+00,  0.7551127E+00,  0.8300018E-01,  0.6409670E-02,  0.1058790E+00], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "mpi_leftright"
nodes = 24
ranks = 96
cores_per_rank = 1
mode = "strong"
rep = 3
experiment = "results_mpi_leftright_strong_1024_n96"
binary = "./stencil2d-mpi.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 24 -n 96 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 15 ./stencil2d-mpi.x --nx 1024 --ny 1024 --nz 64 --num_iter 1024"
env_overrides = "OMP_NUM_THREADS=1"
srun_extra = "--ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 15"
app_extra = ""
git_commit = "6d1f1ad67bd158c18467cf72dcda5ea414627a95"
git_branch = "main"
git_dirty = 0
git_diff = ""
git_status = ""
sweep_start = "2026-08-31T16:56:17+02:00"
sweep_command = "./experiment_mpi_leftright_scaling.sh --mode strong --ranks 96 --repeats 3 --results-dir results_mpi_leftright_strong_1024_n96"
slurm_jobid = "838263"
t_start = "2026-08-31T16:57:40+02:00"
wall_s = 33.682
t_start_approx = 0
metadata_source = "recorded"

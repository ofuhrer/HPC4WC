# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute avg_pack avg_send avg_wait
data = np.array( [ \
[    4, 1024, 1024,   64,    1024,  0.2485246E+02,  0.1113117E+01,  0.2416158E+02,  0.9833893E+00,  0.2386821E+02,  0.5013635E+00,  0.1333127E-01,  0.4684541E+00], \
] )

# ---- run metadata (appended by lib_experiment.sh run_case) ----
strategy = "mpi_leftright"
nodes = 1
ranks = 4
cores_per_rank = 1
mode = "strong"
rep = 1
experiment = "results_strong_1024_detailed_timing"
binary = "./stencil2d-mpi.x"
command = "env OMP_NUM_THREADS=1 srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default -N 1 -n 4 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 15 ./stencil2d-mpi.x --nx 1024 --ny 1024 --nz 64 --num_iter 1024"
env_overrides = "OMP_NUM_THREADS=1"
srun_extra = "--ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores -t 15"
app_extra = ""
git_commit = "5f659f41dc0b2cbd347870eaa2405c47714603f4"
git_branch = "main"
git_dirty = 1
git_diff = "results_strong_1024_detailed_timing/code_state.diff"
git_status = "results_strong_1024_detailed_timing/code_state.status"
sweep_start = "2026-08-31T12:44:25+02:00"
sweep_command = "./experiment_mpi_leftright_scaling.sh --mode strong --ranks 64\\,32\\,16\\,8\\,4\\,2\\,1 --nx 1024 --ny 1024 --nz 64 --num_iter 1024 --repeats 3 --results-dir results_strong_1024_detailed_timing"
slurm_jobid = "837243"
t_start = "2026-08-31T13:10:06+02:00"
wall_s = 40.872
t_start_approx = 0
metadata_source = "recorded"

# TODO

- [ ] `day4/01-GPU-programming-cupy.ipynb`: Bonus Exercise 15 currently cannot be solved because we do not have access to an OpenACC Fortran compiler on Alps.
- [ ] `day4/memory_management.py`: revise the Bonus 14 memory-management material to remove the AI-assistance disclaimer and resolve the still-outstanding expert-review note before publishing.
- [ ] `day2/02-OpenMP-exercises.ipynb`: harden the thread-scaling cells 37 and 40 so they fail clearly when the JupyterHub Slurm allocation has too few CPUs per task, and avoid plotting stale `out.txt` / `out_j.txt` data after failed `srun -c` commands.

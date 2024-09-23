# HPC4WC-Project-13
In this project, the data models NumPy, Numba, Torch and JAX were used
to implement the fourth order diffusion equation over a cuboid domain. The
goal was to compare the data models and to determine the best data model
for diffusion simulations. The code was run on one node of the supercomputer
Piz Daint [1] and to compare the models, the precision (32 or 64 bit) and
the number of cores (1, 6 or 12) were varied. When several cores were used,
Torch had the best performance, due to its highly optimized parallelization
capabilities. However, when only one core was used, JAX performed best for 32
bit precision, while Numba performed best for 64 bit precision. These models
performed well, as both of them use Just In Time (JIT) compilation, allowing
for additional optimization and efficient memory accessing.

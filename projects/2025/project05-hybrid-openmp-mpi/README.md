Plan for hybrid MPI/OPENMP Project 
Best configuration of a hybrid (OpenMP / MPI) weather and climate model: Typical weather and climate models use OpenMP to parallelize in the vertical and MPI to domain-decompose in the horizontal. Implement domain decomposition using MPI in the stencil2d-kparallel.F90 version of the code. Find the optimal configuration of running the code on an Alps node (4 x Grace-Hopper). How many threads and how many MPI ranks give the best performance? Can you understand why?


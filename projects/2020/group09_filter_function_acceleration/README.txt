Project 9 in the HPC4WC summer 2020 course
"A fast nan-sensitive mean filter function"
Authors: Verena Bessenbacher and Ulrike Proske
07 08 2020

This directory contains all the scripts produced in the course of the project. Clean, up to date versions are stored directly in this directory, and old, non-commented scripts that we used for development are stored in old_scripts for completeness.
A short overview of the directory content is given in the following:

# naive python implementation of the generic filter
baseline_explicit.py

# naive fortran implementation of the generic filter
baseline_fortran.f90
# improved fortran implementation of the generic filter
fortran_nocopy.f90
# python script for validating the fortran results
baseline_fortran_check.py

# python script that contains all other generic filters: scipy.ndimage.genericfilter, numba, cython
compare_versions.py
# scripts necessary for the cython functions in compare_versions.py
cython_loop2.c
cython_loop2.html
cython_loop2.pyx
cython_loop.c
cython_loop.html
cython_loop.pyx
setup2.py
setup.py

# implementation of a halo update that could be used for all the generic filter versions
halo_bc_update.py

# python environment
environment.yml

# Source code of the generic filter function
ndimage_sourcecode

# Place for scripts used for development along the way
old_scripts


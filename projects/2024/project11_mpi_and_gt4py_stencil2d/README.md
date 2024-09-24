# Project 11: MPI version of GT4Py code
By Belinda Hotz, Michael Klein, Vishnu Selvakumar & Yijun Wang

Supervised by: Oliver Fuhrer

## Problem statement
*MPI version of GT4Py code: Merge your work from MPI parallelization into the gt4py version and make sure that all code is ported using gt4py (including halo-updates). This code would be able to fully leverage the Piz Daint supercomputer! Analyze and optimize its performance and do a weak and strong scaling investigation on Piz Daint for different backends.*

## Folder structure

### Report
MPI_Version_of_GT4Py_Code.pdf 

### Analysis scripts
analysis_script.ipynb: Contains all analysis and plotting of our report

validation.ipynb: Validates the output of stencil2d-gt4py-mpi-base.py and stencil2d-gt4py-mpi-laplacian.py to stencil2d-gt4py.py by calling the compare_fields.py for different configurations of our new implementations.

job_weak_scaling.sh: Bash-Script to run the stencil on GPU

weak-scaling-plot.py: Script for generating the plots for the weak scaling

### 4 versions of the stencil code:
stencil2d-mpi.py: MPI reference

stencil2d-gt4py.py: GT4py reference

stencil2d-gt4py-mpi-base.py: our MPI-integrated GT4py implementation 

stencil2d-gt4py-mpi-laplacian.py: our modified Laplacian MPI-integrated GT4py implementation 

###  Folders
**data/**: contains the runtimes of the stencil implementations for various backends

**plots/**: contains the plots generated for this report

### Other requirements
partitioner.py: Contains an MPI class

compare_fields.py: Compares the output of one filed to a second one and checks if they are equal
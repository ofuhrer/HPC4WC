
# Group 02 Project: Implementing a Diffusion Stencil on a Sphere using MPI

## Group Members

Shruti Nath & Beat Hubmann

## Project Synopsis

Implement an MPI stencil code on a cubed-sphere grid, allowing to run on a full sphere.
Investigate weak and strong scalability of the code on moderate node counts.

## Running the Code

### Command Line Options

* ```--nx``` Number of gridpoints in x-direction. Typical value: 120
* ```--ny``` Number of gridpoints in y-direction. Typical value: 120
* ```--nz``` Number of gridpoints in z-direction. Typical value: 60
* ```--num_iter``` Number of iterations. Typical value: 1020
* ```--num_halo``` Number of halo-pointers in x- and y-direction. Default value: 2
* ```--plot_result``` Save plots of the result? Default value: False
* ```--verify``` Save verifications plots and output subdomains for up to 8 x/y gridpoints plus 2*2 halo points? No diffusion, overrides num_iter, plot_result options. Default value: False

### Typical Run Command for Production

```mpirun -n 6 python3 ./stencil3d-mpi.py --nx=120 --ny=120 --nz=60 --num_iter=1020 --plot_result=true```

### Typical Run Command for Verification

#### Small Arrays - Get Command Line Array Slices Output plus Verification

```mpirun -n 6 python3 ./stencil3d-mpi.py --nx=8 --ny=8 --nz=60 --num_iter=1020 --verify=true```

#### Regular-sized Arrays - Get Verification

```mpirun -n 6 python3 ./stencil3d-mpi.py --nx=120 --ny=120 --nz=60 --num_iter=1020 --verify=true```


### Typical Plotting Command for Production
```python3 ./super_plotter.py```

## Contents of Submission

### Root Directory

* ```stencil3d-mpi.py``` Main cubed sphere program
* ```cubedspherepartitioner.py``` Contains CubedSpherePartitioner class required for main program
* ```stencil2d-mpi.py``` Slightly modified 2D counterpart to main cubed sphere program for reference
* ```cubedspherepartitioner.py``` Contains Partitioner class required for 2D counterpart main program
* ```super_plotter.py``` Plotting program to assemble individual subdomain plots from MPI workers into unfolded cube plot
* ```requirements.txt``` Python ```pip``` requirements file
* ```LICENSE``` MIT License file
* ```README.md``` This file

### scaling_analysis Directory

* ```submit_strong_scaling.sh``` Slurm submission script for strong scaling analysis runs on CSCS Piz Daint
* ```submit_weak_scaling.sh``` Slurm submission script for weak scaling analysis runs on CSCS Piz Daint
* ```cubed_sphere_strong_scaling.*.out``` Strong scaling analysis run outputs from CSCS Piz Daint
* ```cubed_sphere_weak_scaling.*.out``` Weak scaling analysis run outputs from CSCS Piz Daint

### report Directory

* ```group02_project_report.pdf``` Project report


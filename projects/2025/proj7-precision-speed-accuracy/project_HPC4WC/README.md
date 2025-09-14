# Project: Exploring the Tradeoff Between Resolution and Floating-Point Precision in 2D Burgers’ Equation
**Authors**: Max Kimmlingen, Marco Muccioli, and David Tschan

## Description
The aim of this project is to investigate the tradeoff between floating-point precision and grid resolution in solving the 2D viscous Burgers’ equation. This equation involves both advection and diffusion terms, such that higher grid resolution generally improves the accuracy of the simulation. Similarly, higher floating-point precision is expected to enhance numerical accuracy. 
Since the problem is memory-bound, reducing data transfer is critical for performance. This can be achieved either by reducing the precision or by decreasing the resolution. We will program using C++ as it allows us to test a broad range of floating-point precisions: half precision (f16, could not be implemented), single precision (f32), double precision (f64) and long double precision (f128).
The core objective is to analyze the performance tradeoff between precision and resolution, by comparing a set precision-resolution configurations with a high-resolution high-precision benchmark.
The code for solving the equation is largely based on the stecil2d.cpp code.

**Tested Resolutions**: (128, 128), (256, 256), (512, 512)
**Tested Precisions**: float, double, long double
**Benchmark**: (1024, 1024) with float

## Prerequesites:
To reproduce our results the following folders are necessary: 
    - ./animations
    - ./figures
    - ./input_data
    - ./output_data


## Steps to reproduce our Results:
- create initial conditions: 
    - In make_initial_fields.cpp:
        - specify precision in line 17: float, double, long double
        - adjust line 24 accordingly: "single", "double", "longdouble"
    - for each of these 3 cases: 
        - compile: eg. `mpic++ make_initial_fields.cpp -o make_initial_fields_single -O3`
        - then run: `srun -n 1 ./make_initial_fields --nx 1024 --ny 1024 --nz 1`
    This will generate **all** initial conditions and save them to ./input_data

- run all configurations:
    In 2d_Burger.cpp:
        - specify precision in line 19: float, double, long double
        - adjust line 26 accordingly: "single", "double", "longdouble"
    - for each of these 3 cases:
        - compile: eg. `mpic++ 2d_Burger.cpp -fomp -o 2d_Burger_single -O3`
        - for each resolution in *Tested Resolutions*:
            run: eg. `srun -n 1 ./2d_Burger_single --nx X --ny Y --nz Z --Tend TEND --nums NUMS` with Z=1, TEND=1, NUMS=100
        - for the benchmark:
            run: `srun -n 1 ./2d_Burger_longdouble --nx 1024 --ny 1024 --nz 1 --Tend 1 --nums 100` (needs around 5mins)
    This produces the 3x3 tested configurations and the benchmark. It saves their time series in ./output_data. Additionally, it prints the runtime (shwon in Table 1 of the report) to the terminal. 

- Producing Figure 1 in report:
    run: `srun -n 1 python burger_animation.py --nx 1024 --ny 1024 --nz 1 --Tend 1 --prec longdouble --nums 100`
    This produces a GIF of the benchmark simulation and saves it in ./animations . The first and last frame of this GIF are shown in Figure 1.

- Producing Figure 2: 
    run: `srun -n 1 python visualisation.py --nx 512 --ny 512 --nz 1 --Tend 1 --nums 100 --mode u`
    This produces all the relevant images and saves them in ./figures .

- Producing Figure 3:
    run `srun -n 1 python l2_and_psd_error.py`
    This produces the plot and saves it in ./figures .


## Comments on Files: 
### Files used for project results:
- make_initial_fields.cpp
    See above. This file generates initial conditions in a specified precison (command line arg) for resolutions (nx,ny), (nx/2,ny/2), (nx/4,ny/4), and (nx/8,ny/8) where nx,ny are command line args.
    Lines 08-85, 100-111, and 116-119 can be commented out to produce only the initial condition for (nx,ny).
- In 2d_Burger.cpp
    See above. This file takes the initial condition of a specified precision and resolution and integrates forward in time. It saves a specified number (nums) of snapshots of the simulation.
- burger_animation.py
    See above. The resolution and precision can be changed to produce simulations of the other configurations.
- visualisation.py
    See above. For a specified resolution (command line arg) nx,nx output files for (nx,ny), (nx/2,ny/2), (nx/4,ny/4) need to exist.
- l2_and_psd_error
    See above. The benchmark and tested configurations are hardcoded in the main part of the code and can be changed as needed.
- utils_Burger.h
    Utils for solving the Burger equation.
- utils.h
    Utils from the lecture, modified.

### Others
- interpolate_initial_fields.py
    Allows to interpolate a given initial condition to any desired resolution for when make_initial_fields.cpp is to inflexible. 
    The field to interpolate as well as the resolutions to interpolate to are hardcoded in the main part of the code and can be changed as needed.
- stencil2d-base.cpp
    Base code from the lecture. 
- stencil2d.cpp
    Intermediate step toward final 2d_Burger.cpp.
- data_to_parquet.py
    A group member had issues with saving different data types.

### Outdated
- project.sh
- analysis.py
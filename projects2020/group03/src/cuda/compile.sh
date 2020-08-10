#!/bin/bash
g++              -O3 -Iinclude src/update_halo.cpp src/apply_stencil_cpu.cpp                      src/stencil2d_cpu.cpp              -fopenmp -o stencil2d_cpu.x
nvcc -arch=sm_60 -O3 -Iinclude src/update_halo.cpp                           src/apply_stencil.cu src/stencil2d_gpu.cu    -Xcompiler -fopenmp -o stencil2d_gpu.x
nvcc -arch=sm_60 -O3 -Iinclude src/update_halo.cpp src/apply_stencil_cpu.cpp src/apply_stencil.cu src/stencil2d_hybrid.cu -Xcompiler -fopenmp -o stencil2d_hybrid.x

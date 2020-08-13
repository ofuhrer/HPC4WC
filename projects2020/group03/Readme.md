# Overlapping computation on the CPU and the GPU

## Description

> Implement a version of `stencil2d.F90` that offload part of the work onto the GPU and keeps parts on the CPU. How does domain decomposition affect this? Extend the stencil to 3d and decide on domain decomposition in this case. If time permits try any dense linear algebra method. test if we can get an advantage of doing a part of the work on the CPU while the GPU kernel is running. (Literatur [1])

Final code versions:
 - CPU (OpenMP): `src/cuda/stencil2d_cpu.cpp`
 - GPU (OpenMP): `src/openmp_target/diffusion_openmp_target2.f`
 - GPU (OpenACC):
 - GPU (CUDA): `src/cuda/stencil_2d_gpu.cu`

[1] https://reader.elsevier.com/reader/sd/pii/S0045782511000235?token=D6719774F918FDC2DBA99AF121BCA0F1F371703BACD747EDFFDC2771E43E920816D31843CEAF2AF698CD81FC86624D65
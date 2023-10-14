### High Performance Computing for Weather and Climate

| Authors | Thomas Liniger & Faveo Hörold |
| --- | --- |
| **Advisors** | Oliver Fuhrer & Tobias Wicky |
| **Semester** | Spring Semester 2023 |
| **Project** | Work Project 10 |
| **Title** | Abstracted and Low-Level GPU Programming Comparison |
| **Date** | 27.08.2022 |

**Task Description:** Implement a CUDA and/or OpenACC version of a simplified version of stencil2d (no periodic BCs). Validate your results. Compare performance against the CuPy and Fortran versions of the code for a single CPU or GPU socket. If time permits, analyze and optimize the performance of the implementations.

**Abstract:** We implement a number of CUDA kernels to solve a fourth-order diffusion equation. We benchmark each implementation on multiple CUDA devices and relate the performance of each implementation and optimization to the characteristics of the device. We achieve an order-of magnitude performance improvement over a reference Fortran implementation of the code, accelerated with OpenACC. We also investigate the performance of a shared memory implementation and make a fined-grained comparison against our normal implementation using the Nvidia profiler.

# High-Performance Computing for Weather and Climate (HPC4WC)

[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue?style=for-the-badge&labelColor=4c4c4c)](LICENSE)
[![Material checks](https://img.shields.io/github/actions/workflow/status/ofuhrer/HPC4WC/material.yml?branch=main&label=Tests&style=for-the-badge&labelColor=4c4c4c)](https://github.com/ofuhrer/HPC4WC/actions/workflows/material.yml)

High Performance Computing for Weather and Climate is a hands-on course on
performance-aware scientific programming. The material uses progressively more
complex versions of stencil computations to introduce hardware concepts,
parallel programming models, and performance-portable abstractions used in
weather and climate applications.

The course material is organized by day:

- [Day 1](day1): single-core performance, stencil programs, performance metrics,
  memory hierarchy, memory bandwidth, peak floating-point performance,
  arithmetic intensity, roofline model, array storage in memory, and
  data-locality optimizations such as blocking, fusion, and inlining.
- [Day 2](day2): shared-memory parallelism with OpenMP, speedup, Amdahl's law,
  parallelization, synchronization, and variable scoping.
- [Day 3](day3): distributed-memory parallelism with MPI, message passing,
  point-to-point communication, deadlock, non-blocking communication,
  gather/scatter operations, domain decomposition, halo points, and halo updates.
- [Day 4](day4): GPU programming, hybrid node architecture, high-level GPU
  programming with CuPy, data management, synchronization, vectorization,
  platform-agnostic code, and lower-level GPU programming.
- [Day 5](day5): high-level programming models, domain-specific languages,
  GT4Py, performance portability, and abstractions for stencil computations.

## Running Checks

Some course exercises require specific HPC hardware, compilers, MPI, GPUs, or a
CSCS/Alps environment. The repository also contains lightweight checks that can
run locally and in GitHub Actions without those systems.

The current automated material check covers Day 1, Day 2, and Day 5:

```bash
python tools/check_material.py day1 day2 day5
```

This validates notebooks and course files, checks Python and shell syntax,
compiles selected Fortran/C++ material when suitable compilers are available,
verifies local notebook image references, and runs small smoke tests for selected
scripts. The check is intentionally lightweight; full performance runs and
GPU/cluster-specific exercises should still be validated in the appropriate
course environment.

Generated solution bundles in `day*/solutions/` are ignored while preparing the
course. Generate them locally only when the solutions should be published.

## Pre-commit Checks

The repository provides conservative pre-commit checks for course preparation:

```bash
python -m pip install pre-commit
pre-commit install
```

The hooks block generated solution bundles from being committed, verify that
generated student material is current for days that use generated student
material, and check formatting for Python tooling in `tools/`. They
intentionally do not auto-format notebooks or Fortran course material.

# GT4Py vs hand-optimized Fortran

    domain              256 x 256 x 64
    num_iter            128
    initial condition   box
    CPU runs            one Grace core, taskset -c 0
    GPU runs            one H100
    Fortran reference   stencil2d-filter-inline, --branchfree when limited
    GT4Py               1.1.10, gtfn backends, float32

    correctness         bit-exact vs Fortran, max |diff| 0,
                        n = 2,4,6,8 x (limiter on, off)

## Wall time per 128 iterations [s]

| n | Limiter | Fortran, 1 core | GT4Py CPU, 1 core | GT4Py GPU, 1 H100 |
|---|---|---|---|---|
| 4 | off | 0.455 | 1.87 (4.1x slower) | 0.0142 (32x faster) |
| 4 | on | 0.966 | 2.48 (2.6x slower) | 0.0144 (67x faster) |
| 8 | off | 0.679 | 3.86 (5.7x slower) | 0.0167 (41x faster) |
| 8 | on | 1.195 | 3.96 (3.3x slower) | 0.0158 (76x faster) |

    factors             relative to the Fortran column, i.e. one GPU
                        against one CPU core
    GPU limiter cost    14.4 ms on vs 14.2 ms off at n=4, within run-to-run
                        variation
    GPU timing includes per-iteration halo update, device sync either side
    lines of code       GT4Py 276 incl. driver and generator;
                        stencil definition 39; generator 60;
                        Fortran 612 to 750 per variant incl. MPI driver
    order-genericity    by regeneration: template emits the 8
                        (order x limiter) programs
    pre-sweeps          separate program statements over per-sweep domains,
                        so intermediates stay full-size 3D

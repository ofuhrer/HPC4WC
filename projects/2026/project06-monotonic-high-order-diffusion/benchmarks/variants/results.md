# Optimization ladder

    domain scan         constant work, nx^2 x num_iter fixed, nx = ny
    nz                  64
    initial condition   box
    alpha_n             1/8^(n/2)
    MPI ranks           72
    statistic           median of 2
    normalization       ms per 1024 iterations
    parser              tools/parse_bench.py

    filter              stencil2d-filter, full-size 3D temporaries
    kblock              stencil2d-filter-kblock, 2D per-k planes
    inline              stencil2d-filter-inline, scalar fluxes, fused loop
    branchy             data-dependent IF
    branch-free         merge intrinsic, --branchfree

| nx = ny | n | Variant | Limiter off [ms] | Branchy [ms] | Branch-free [ms] |
|---|---|---|---|---|---|
| 128 | 4 | filter | 62.4 | 86.6 | not implemented |
| 128 | 4 | kblock | 58.8 | 88.3 | 71.5 |
| 128 | 4 | inline | 55.5 | 102.6 | 72.0 |
| 512 | 4 | filter | 556.1 | 826.2 | not implemented |
| 512 | 4 | kblock | 334.0 | 719.0 | 484.5 |
| 512 | 4 | inline | 276.6 | 977.4 | 506.9 |
| 1024 | 4 | filter | 4644.9 | 5171.5 | not implemented |
| 1024 | 4 | kblock | 1659.6 | 3174.8 | 2190.5 |
| 1024 | 4 | inline | 1400.5 | 3935.0 | 2095.8 |
| 1024 | 8 | filter | 6171.3 | 6848.9 | not implemented |
| 1024 | 8 | kblock | 2250.4 | 3788.5 | 2801.1 |
| 1024 | 8 | inline | 1991.2 | 4435.7 | 2617.5 |

    speedup over filter, off, nx=1024   kblock 2.80x (n=4), 2.74x (n=8)
                                        inline 3.32x (n=4), 3.10x (n=8)
    spread at nx=128                    all variants within ~10%
    branch-free vs branchy, nx=1024 n=4 -31% (kblock), -47% (inline)
    best limited vs filter limited      n=4  5171.5 -> 2095.8  (2.47x)
                                        n=8  6848.9 -> 2617.5  (2.62x)
    limiter overhead, nx=1024 n=4       filter +11%
                                        kblock +91% branchy, +32% branch-free

## Hardware counters

    domain              1024 x 1024 x 64
    filter order n      4
    num_iter            256
    MPI ranks           72
    DRAM traffic        LLC-load-misses x 64 B (overcounts, prefetch)
    arithmetic intensity analytic flop counts: ~12 flop/point/iter off,
                        ~16 on
    nominal DRAM roof   ~500 GB/s

| Variant | Limiter | Wall time [s] | DRAM traffic [GB] | Bandwidth [GB/s] | Arithmetic intensity [flop/B] |
|---|---|---|---|---|---|
| filter | off | 1.125 | 641 | 570 | 0.32 |
| filter | branchy | 1.224 | 723 | 591 | 0.38 |
| kblock | off | 0.382 | 87 | 228 | 2.4 |
| kblock | branchy | 0.767 | 83 | 108 | 3.3 |
| kblock | branch-free | 0.509 | 83 | 162 | 3.3 |
| inline | off | 0.311 | 78 | 251 | 2.7 |
| inline | branchy | 0.954 | 63 | 66 | 4.4 |
| inline | branch-free | 0.492 | 62 | 125 | 4.5 |

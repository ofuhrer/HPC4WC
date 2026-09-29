# Zalesak flux correction vs the simple limiter

    program             src/filter/stencil2d-filter-zalesak.F90
    layout              naive, matching stencil2d-filter (like for like)

    validation          checks 12 to 14 of tools/validate.sh
    C forced to 1       matches unlimited filter to 3.6e-7 (SP)
    C forced to 0       bit-exact vs filter --order 2 --alpha 0.125
    normal operation    monotone at n = 4 and 6

## Runtime

    filter order n      4
    nz                  64
    MPI ranks           72
    statistic           median of 3
    normalization       ms per 1024 iterations

| nx = ny | Unlimited [ms] | Simple limiter [ms] | Zalesak [ms] | Ratio |
|---|---|---|---|---|
| 128 | 66.3 | 95.9 (+45%) | 179.6 (+171%) | 1.87x |
| 1024 | 4637 | 5241 (+13%) | 11650 (+151%) | 2.22x |

## Solution quality

    domain              128 x 128, n = 4, 1 MPI rank
    num_iter            100 from box
    produced by         tools/make_figures.py

    max pointwise diff  2.3e-2
    L2 norm of diff     1.1e-3
    phi=0.5 half-width  axis 32.00, diagonal 44.00, both schemes,
                        0.25 grid-point sampling
    bounds, simple      [0, 1] exact
    bounds, Zalesak     min -1.2e-8 (SP roundoff)
    mean C              0.156 (nx=128, 1024 iters)
                        0.003 (nx=1024, 16 iters)
                        0.0103 (nx=1024, 256 iters)
    contours            fig-zalesak.pdf

## Hardware counters

    domain              1024 x 1024, n = 4
    num_iter            256
    MPI ranks           72
    protocol            as benchmarks/variants/counters.sh

| Variant | Limiter | Wall time [s] | DRAM traffic [GB] | Bandwidth [GB/s] | Arithmetic intensity [flop/B] |
|---|---|---|---|---|---|
| filter | off | 1.156 | 633 | 548 | 0.33 |
| filter | branchy | 1.268 | 721 | 569 | 0.38 |
| zalesak | Zalesak | 2.873 | 2102 | 732 | 0.31 |

    Zalesak flop count  Laplacian pre-sweep 5, corrective flux and
                        gradient 10, provisional phi* 5, ratio factors 12,
                        C application 2, update 4; total ~38 flop per
                        point per iteration
    total flop          652.8 Gflop, 227 Gflop/s attained
    vs limited filter   2.9x DRAM traffic for 2.27x wall time
    732 GB/s > roof     proxy overcounts most here: 9 streamed full-size
                        3D arrays; analytic compulsory traffic ~1.7 TB

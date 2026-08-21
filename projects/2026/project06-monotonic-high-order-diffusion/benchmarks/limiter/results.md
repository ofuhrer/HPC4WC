# Cost of the flux limiter, filter order n = 4

    domain              128 x 128 x 64
    num_iter            1024
    initial condition   box
    alpha               1/32
    MPI ranks           72
    statistic           median of 3

| Program variant | Runtime [ms] | Relative |
|---|---|---|
| `stencil2d-base`, Laplacian of Laplacian | 59.3 | baseline |
| `stencil2d-limiter`, flux form, limiter off | 62.5 | +5.4% |
| `stencil2d-limiter`, flux form, limiter on | 82.6 to 84.8 | +39% to +43% |

    limiter activity            10.6% of nonzero interface fluxes zeroed
    bounds, limiter on          [0, 1] exact
    bounds, limiter off         overshoot 0.10, undershoot 0.05
    flux form vs base, off      max |diff| 2.4e-7 (SP reassociation)
    activity, Xue (2000)        below 1% for smooth convection

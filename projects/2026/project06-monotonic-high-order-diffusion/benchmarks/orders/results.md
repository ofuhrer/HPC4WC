# Runtime and limiter cost vs filter order

    domain              128 x 128 x 64
    num_iter            1024
    initial condition   box
    alpha_n             1/8^(n/2)
    MPI ranks           72
    statistic           median of 3
    program             stencil2d-filter, naive layout

| Filter order n | num_halo | Runtime, limiter off [ms] | Runtime, limiter on [ms] | Limiter overhead | Nonzero fluxes zeroed |
|---|---|---|---|---|---|
| 2 | 1 | 33.5 | 55.5 | +66% | 1.4e-4 |
| 4 | 2 | 63.0 | 88.8 | +41% | 9.8e-2 |
| 6 | 3 | 82.3 | 108.3 | +32% | 4.5e-2 |
| 8 | 4 | 100.0 | 119.7 | +20% | 5.6e-2 |

    absolute limiter cost       +22 to +26 ms, independent of n
    n = 2 zeroed fraction       1.4e-4, not 0: gradients below ~1e-19
                                underflow the SP flux-gradient product,
                                which Eq. 10 (sign(0)=0) then zeroes
    measured vs analytic        lambda(k)^N to ~1e-8 relative, all n

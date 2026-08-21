# Files

Provenance: [c] course code, [c+] course code we extended, everything
else written for this project.

## src/
    Makefile                      per-variant build, VPATH over the dirs below

    baseline/m_utils.F90          timers, field I/O [c]
    baseline/m_partitioner.F90    2D decomposition, scatter/gather [c]
    baseline/stencil2d-base.F90   day-3 MPI, runtime halo width [c+]

    filter/stencil2d-limiter.F90        4th order in flux form, Xue Eq. 10
    filter/stencil2d-filter.F90         order-generic n = 2,4,6,8, naive
    filter/stencil2d-filter-kblock.F90  temporaries as 2D per-k planes
    filter/stencil2d-filter-inline.F90  scalar fluxes, one fused loop
    filter/stencil2d-filter-zalesak.F90 Zalesak flux correction

    gt4py/stencil2d-filter-gt4py.py     GT4Py version and generator
    tests/test_scatter_gather.F90       partitioner round-trip

## tools/
    validate.sh              the 14-check suite
    check_amplification.py   measured vs analytic damping
    check_monotonic.py       bounds check, both directions
    read_field.py            halo-aware field read and compare
    parse_bench.py           raw logs to median tables
    make_figures.py          regenerates the figures
    perf_wrap.sh             per-rank perf stat, ARM PMU

## benchmarks/
One directory per experiment; `results.md` holds the parsed medians.

    limiter/    cost of the limiter at n=4
    orders/     cost vs filter order
    variants/   optimization ladder, plus counters.sh for the roofline
    gt4py/      GT4Py vs Fortran, CPU and GPU
    zalesak/    Zalesak cost, quality and counters

## root
    README.md         build, test, layout
    AI_STATEMENT.md   declaration on the use of generative AI
    LICENSE           GPL-3.0, inherited from the course code

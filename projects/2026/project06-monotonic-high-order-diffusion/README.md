# Monotonic High-Order Diffusion on NVIDIA Grace

HPC4WC (701-1270-00L), ETH Zurich, FS26, project 6.
Stefanie Boersig, Boaz Ko, Ben Bullinger. Advisor: Max Harvey.

The flux limiter of Xue (2000) in the course's `stencil2d`, for filter
orders n = 2, 4, 6, 8, with an optimization ladder and a GT4Py port.

## Test

Needs `mpif90`, make, and Python 3 with NumPy.

    tools/validate.sh

14 checks. Exits non-zero on the first failure; prints
`== all validations passed` on success.

## Build and run

    cd src
    make allversions
    mpirun -n 4 ./stencil2d-filter.x --nx 128 --ny 128 --nz 64 \
        --num_iter 1024 --order 6 --limiter

Options: `--order {2,4,6,8}`, `--limiter`, `--branchfree` (kblock and
inline only), `--alpha`, `--num_halo`, `--ic {box,mode}`.

## Layout

    src/          baseline/ filter/ gt4py/ tests/
    tools/        validate.sh and the analysis tooling
    benchmarks/   one directory per experiment
    INDEX.md      every file, one line each

`tools/make_figures.py` (needs matplotlib) writes to `figures/`; set
`FIGDIR` to write elsewhere.

## Attribution

Stefanie Boersig <stefanie.boersig@env.ethz.ch>,
Boaz Ko <boazko@student.ethz.ch>,
Ben Bullinger <ben.bullinger@inf.ethz.ch>.

`src/baseline/` is course code by O. Fuhrer and coauthors,
github.com/ofuhrer/HPC4WC, and keeps its original headers.
On the use of AI: `AI_STATEMENT.md`. Licensed under GPL-3.0 as a
derivative of the course code: `LICENSE`.

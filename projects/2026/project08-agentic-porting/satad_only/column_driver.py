#!/usr/bin/env python
"""Single-column driver for the GT4Py saturation-adjustment port.

The Python analogue of ``../fortran/column_driver.f90``: it reads a 1-D
atmospheric column from a CSV, calls the GT4Py ``satad`` once (via
``satad_numpy``), prints the per-level change, and writes the updated column
back out. Same CSV contract as the Fortran driver -- one header row then ``nlev``
data rows, columns ``rho,tk,qv,qc`` -- so the two drivers' outputs are directly
comparable.

Usage
-----
    column_driver.py [in_csv [out_csv [tol [maxiter]]]]

All arguments are optional and default (relative to this file) to
``example/fields.csv``, ``example/output_fields_gt4py.csv``, ``tol=1e-3``,
``maxiter=10``. Unlike the Fortran driver, ``maxiter`` is *fixed* at
``MAXITER`` (= 10) because the GT4Py Newton loop is unrolled that many times at
import; passing any other value is rejected rather than silently ignored.

Run it (from anywhere) with:  ``uv run python -m satad_only.column_driver``
"""

from __future__ import annotations

import argparse

import numpy as np

from satad_only import EXAMPLE_DIR
from satad_only.satad_gt4py import DEFAULT_TOL, MAXITER, satad_numpy

DEFAULT_INPUT = EXAMPLE_DIR / "fields.csv"
DEFAULT_OUTPUT = EXAMPLE_DIR / "output_fields_gt4py.csv"

FIELD_HEADER = "rho,tk,qv,qc"


def read_fields(path):
    """Read a ``rho,tk,qv,qc`` CSV (one header row) into four 1-D arrays."""
    data = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def write_fields(path, rho, tk, qv, qc):
    """Write ``rho,tk,qv,qc`` at full float64 precision (~17 sig figs)."""
    arr = np.column_stack([rho, tk, qv, qc]).astype(np.float64)
    np.savetxt(path, arr, delimiter=",", header=FIELD_HEADER, comments="",
               fmt="%.16E")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("in_csv", nargs="?", default=DEFAULT_INPUT,
                        help="input column CSV (default: example/fields.csv)")
    parser.add_argument("out_csv", nargs="?", default=DEFAULT_OUTPUT,
                        help="output CSV (default: example/output_fields_gt4py.csv)")
    parser.add_argument("tol", nargs="?", type=float, default=DEFAULT_TOL,
                        help=f"temperature tolerance [K] (default: {DEFAULT_TOL})")
    parser.add_argument("maxiter", nargs="?", type=int, default=MAXITER,
                        help=f"Newton iterations; fixed at {MAXITER} in GT4Py")
    args = parser.parse_args(argv)

    if args.maxiter != MAXITER:
        parser.error(
            f"maxiter is compile-time fixed at {MAXITER} in the GT4Py port "
            f"(the Newton loop is unrolled); got {args.maxiter}."
        )

    rho, tk, qv, qc = read_fields(args.in_csv)
    nlev = tk.size
    print(f"column_driver (gt4py): nlev = {nlev}")

    tk_out, qv_out, qc_out = satad_numpy(rho, tk, qv, qc, tol=args.tol)

    print("column_driver (gt4py): per-level change from saturation adjustment")
    print("  lev        dT [K]        dqv [kg/kg]        dqc [kg/kg]")
    for k in range(nlev):
        print(f"{k + 1:5d}{tk_out[k] - tk[k]:18.8E}"
              f"{qv_out[k] - qv[k]:18.8E}{qc_out[k] - qc[k]:18.8E}")

    write_fields(args.out_csv, rho, tk_out, qv_out, qc_out)
    print(f"column_driver (gt4py): wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

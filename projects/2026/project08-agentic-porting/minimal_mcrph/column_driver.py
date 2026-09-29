#!/usr/bin/env python
"""Command-line single-column driver for the GT4Py minimal two-moment port.

The Python analogue of ``fortran/column_driver.f90``, and the counterpart of
``satad_only/column_driver.py``. It holds no physics and no CSV format
knowledge of its own: the timestep is :class:`minimal_mcrph.driver.Driver`, the
column layout is :mod:`minimal_mcrph.csv_io`, and what is left here is the
argument parsing, the run summary, and the two calls that join them.

Read ``fields.csv`` plus its half-level heights from ``hhl.csv``, run one
timestep, print what that timestep did to the column, write the updated column
back out. Same CSV contract as the Fortran driver -- the 23 columns of
``csv_io.COLUMNS``, one header row then ``nlev`` data rows -- so the two
drivers' outputs are directly comparable.

The summary is in two parts, because 17 evolving fields will not fit in one
per-level table the way saturation adjustment's three do:

* a per-level table of ``dtk``, ``dqv``, ``dqc``, ``dqi`` -- the four the scheme
  moves in essentially every column;
* a per-field roll-up of the largest change anywhere in the column, over every
  field the timestep can touch, so a species that moved outside those four is
  still visible.

Unlike the Fortran driver there are no surface precipitation rates to report:
sedimentation is exactly what this variant leaves out, so they would be zero.

Usage
-----
    column_driver.py [in_csv [out_csv [hhl_csv [dt]]]] [--backend ...]

The positionals are the Fortran driver's, in its order, and all are optional;
they default (relative to this file) to ``example/fields.csv``,
``example/output_fields_gt4py.csv``, ``example/hhl.csv`` and ``dt=30.0``. The
output name differs from the Fortran's ``output_fields.csv`` on purpose, so a
port run and a ``make run`` in ``fortran/`` can sit side by side in
``example/`` instead of overwriting each other.

Run it (from anywhere) with:  ``uv run python -m minimal_mcrph.column_driver``
"""

from __future__ import annotations

import argparse
import dataclasses

import gt4py.next as gtx
import numpy as np

from minimal_mcrph import EXAMPLE_DIR
from minimal_mcrph.csv_io import read_fields_csv, write_fields_csv
from minimal_mcrph.driver import ColumnState, Driver

DEFAULT_INPUT = EXAMPLE_DIR / "fields.csv"
DEFAULT_OUTPUT = EXAMPLE_DIR / "output_fields_gt4py.csv"
DEFAULT_HHL = EXAMPLE_DIR / "hhl.csv"
DEFAULT_DT = 30.0  # column_driver.f90's default

# Public so anything else offering a backend switch (scripts/compare_to_reference.py)
# spells the choices the same way rather than keeping its own copy.
BACKENDS = {"embedded": None, "gtfn_cpu": gtx.gtfn_cpu}

# What `run_timestep` hands back unchanged: geometry, the thermodynamic state it reads
# but never writes, and ninagi (which only moves under luse_agi, off for this config).
# Everything else in ColumnState is something the timestep can change -- deriving the
# list that way means a new field shows up in the summary without being added here.
_PASSTHROUGH = ("hhl", "rho", "pres", "w", "ninagi")
EVOLVED_FIELDS = tuple(
    f.name for f in dataclasses.fields(ColumnState) if f.name not in _PASSTHROUGH
)

# The subset printed level by level: the thermodynamic pair plus the two condensate
# species the bundled column actually exercises.
PER_LEVEL_FIELDS = ("tk", "qv", "qc", "qi")


def print_per_level(before: ColumnState, after: ColumnState) -> None:
    """Per-level change in the four headline fields, one row per model level."""
    print("column_driver (gt4py): per-level change over one timestep")
    print(f"{'lev':>5}" + "".join(f"{'d' + name:>18}" for name in PER_LEVEL_FIELDS))
    for k in range(before.rho.size):
        deltas = "".join(
            f"{getattr(after, name)[k] - getattr(before, name)[k]:18.8E}"
            for name in PER_LEVEL_FIELDS
        )
        print(f"{k + 1:5d}{deltas}")


def print_field_summary(before: ColumnState, after: ColumnState) -> None:
    """Largest change anywhere in the column, for every field the timestep touches."""
    print("column_driver (gt4py): largest change per field over the column")
    print(f"{'field':<10}{'max |delta|':>16}{'at lev':>8}{'before':>18}{'after':>18}")
    for name in EVOLVED_FIELDS:
        old, new = np.asarray(getattr(before, name)), np.asarray(getattr(after, name))
        k = int(np.argmax(np.abs(new - old)))
        print(f"{name:<10}{abs(new[k] - old[k]):16.3E}{k + 1:8d}{old[k]:18.8E}{new[k]:18.8E}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("in_csv", nargs="?", default=DEFAULT_INPUT,
                        help="input column CSV (default: example/fields.csv)")
    parser.add_argument("out_csv", nargs="?", default=DEFAULT_OUTPUT,
                        help="output CSV (default: example/output_fields_gt4py.csv)")
    parser.add_argument("hhl_csv", nargs="?", default=DEFAULT_HHL,
                        help="half-level heights CSV (default: example/hhl.csv)")
    parser.add_argument("dt", nargs="?", type=float, default=DEFAULT_DT,
                        help=f"timestep [s] (default: {DEFAULT_DT})")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="embedded",
                        help="GT4Py backend (default: embedded; gtfn_cpu compiles first)")
    args = parser.parse_args(argv)

    state, ssat, qrsflux = read_fields_csv(args.in_csv, args.hhl_csv)
    print(
        f"column_driver (gt4py): nlev = {state.rho.size}, dt = {args.dt} s, "
        f"backend = {args.backend}"
    )

    result = Driver(backend=BACKENDS[args.backend]).run_timestep(state, dt=args.dt)

    print_per_level(state, result)
    print_field_summary(state, result)

    write_fields_csv(args.out_csv, result, ssat, qrsflux)
    print(f"column_driver (gt4py): wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

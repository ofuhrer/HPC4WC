# Minimal 2-Moment Microphysics Port

This directory contains a stripped-down version of the ICON 2-moment microphysics scheme, ported to GT4Py.

## Structure

This directory *is* the `minimal_mcrph` Python package — the GT4Py port sits at its
top level, beside the Fortran it was ported from. `FORTRAN_DIR` and `EXAMPLE_DIR`
(defined in `__init__.py`, anchored to the file rather than to the working
directory) are how the tests and scripts find the reference code and the sample
column, so everything below runs identically from any working directory.

```
minimal_mcrph/
├── fortran/              # Original Fortran code + column_driver.f90
├── example/              # Shared sample column: fields.csv, hhl.csv,
│                         #   output_fields.csv (Fortran output, gitignored),
│                         #   output_fields_gt4py.csv (port output, gitignored)
├── __init__.py
├── config.py
├── constants.py
├── particles.py
├── ccn_activation.py
├── csv_io.py
├── driver.py
├── column_driver.py      # CLI wrapper around driver.py, mirrors column_driver.f90
├── data/
│   └── ccn_otab.npz
├── stencils/
│   ├── __init__.py
│   ├── satad.py
│   ├── saturation.py
│   ├── processes.py
│   ├── vapor_deposition.py
│   ├── ice_nucleation.py
│   ├── particle_helpers.py
│   ├── unit_conversion.py
│   ├── housekeeping.py
│   └── post_housekeeping.py
├── tests/
│   ├── unit_tests/
│   └── integration_tests/
├── scripts/              # compare_to_reference.py, extract_ccn_otab.py
└── PORT_NOTES.md         # notes on the port itself
```

## Components

### Saturation Adjustment
The saturation adjustment subroutine from ICON's `mo_satad.f90`. Adjusts temperature and humidity to near saturation.

### Microphysical Processes
GT4Py field operators for all microphysical processes in the minimal 2-moment scheme:
- CCN activation (cloud water formation)
- Homogeneous/heterogeneous ice nucleation
- Cloud droplet freezing
- Vapor deposition / condensation relaxation
- Ice crystal melting
- Default number concentration initialization

### Driver
The `driver.py` module contains the main driver class that orchestrates the saturation adjustment and microphysical processes. It manages:
- Initialization of lookup tables and particle coefficients
- Ordering of process calls (typically: satad → microphysics → satad)
- Field allocation and domain management

`column_driver.py` is the command-line wrapper around it, and the Python counterpart of
`fortran/column_driver.f90` — it holds no physics and no CSV layout of its own, only the
argument parsing, the run summary and the two calls that join `driver.py` to `csv_io.py`.

## Running one timestep

```bash
uv run python -m minimal_mcrph.column_driver                      # bundled column, dt=30
uv run python -m minimal_mcrph.column_driver --backend gtfn_cpu   # compiled (slow first run)
```

It reads `example/fields.csv` + `example/hhl.csv`, runs one timestep, prints the summary
below, and writes the updated column to `example/output_fields_gt4py.csv` at full float64
precision — beside, not over, the Fortran's `output_fields.csv`, so the two can be diffed.

The summary comes in two parts, because 17 evolving fields do not fit in one per-level
table the way saturation adjustment's three do: a per-level table of `dtk`, `dqv`, `dqc`,
`dqi`, then a per-field roll-up of the largest change anywhere in the column, so a species
that moved outside those four is still visible. There are no surface precipitation rates
to report the way the Fortran driver does — sedimentation is what this variant leaves out.

Positional arguments are the Fortran driver's, in its order, all optional:

```bash
uv run python -m minimal_mcrph.column_driver in.csv out.csv hhl.csv 60.0
```

## Validation

The reference is the compiled Fortran, built and run during the test session. Nothing
in the suite compares against a committed CSV or against numbers pasted from an earlier
run, so a change to `fortran/` is picked up automatically instead of silently
invalidating a stale expectation.

**The port currently reproduces the Fortran to 1.6e-15 relative (about 7 ULP) on every
field, at every process boundary, across all six test columns**; the `warm` column comes
out bit-identical. Tests assert at `rtol=1e-14` — see `tests/conftest.py`, where every
tolerance is annotated with the measurement that justifies it.

Three layers, each catching something the others cannot:

| test | what it holds |
| --- | --- |
| `integration_tests/test_full_column.py` | the whole timestep, per scenario and backend, plus oracle-free invariants (water conservation, non-negativity, mean masses in range) |
| `unit_tests/test_stage_boundaries.py` | the column at each of the nine process boundaries, so a discrepancy localises to a process rather than to "somewhere" |
| `test_scenario_sanity.py` | that each scenario column really is the regime its name claims, checked with independent (Murphy-Koop) thermodynamics in `mcrph_common/thermo.py` |

The scenarios exist because the bundled column alone exercises surprisingly little: it
has `w == 0` at every level (so CCN activation and homogeneous freezing never fire) and
no snow, graupel or hail (so three of four deposition species are inert). `updraft`,
`mixed_species`, `warm`, `deep_cold` and `mixed_random` open those paths.

```bash
uv run pytest minimal_mcrph/tests -v                  # everything
uv run pytest minimal_mcrph/tests -k "not gtfn" -v    # fast loop; skips compilation
```

`gtfn_cpu` compilation dominates the runtime and can take many minutes for the larger
stencils. `-k "not gtfn"` runs the same physics through embedded execution in seconds.

### Looking at the numbers by hand

```bash
uv run python minimal_mcrph/scripts/compare_to_reference.py                    # bundled column
uv run python minimal_mcrph/scripts/compare_to_reference.py --scenario all
uv run python minimal_mcrph/scripts/compare_to_reference.py --stages           # per boundary
```

### Per-stage dumps from the Fortran

`fortran/mo_stage_dump.f90` writes the column state at each process boundary at 17
significant digits, when — and only when — the driver is given a fifth argument. A bare
`make run` is byte-for-byte unaffected.

```bash
cd minimal_mcrph/fortran
./build/column_driver ../example/fields.csv /tmp/out.csv ../example/hhl.csv 30.0 /tmp/stages
```

### The reference column, and regenerating it

`example/` holds four files, and they are not all the same kind of thing:

| file | tracked in git? | what it is |
| --- | --- | --- |
| `fields.csv` | yes | the input column: 23 columns, one row per model level |
| `hhl.csv` | yes | nlev+1 half-level heights, model top first |
| `output_fields.csv` | **no** | the Fortran's output for that column, a build product |
| `output_fields_gt4py.csv` | **no** | the port's output for that column, a build product |

Both outputs are build products, written by running their respective drivers — the
Fortran's with `make run` in `fortran/`, the port's with `uv run python -m
minimal_mcrph.column_driver`.

Both are matched by the `**/output_fields*.csv` rules in the repo-root `.gitignore`, so
**a fresh clone has neither**. Regenerate the Fortran's by running the Fortran:

```bash
cd minimal_mcrph/fortran
make run                      # builds build/column_driver, runs it, writes ../example/output_fields.csv
```

`make run` also tees the driver's stdout — including the surface precipitation rates and
the scheme's own startup messages — to `fortran/logs/driver_run.log`.

**You do not need to do this before running the tests.** The suite builds and runs the
Fortran itself, for every scenario, during the session; it never reads
`output_fields.csv`. A fresh clone can go straight to `uv run pytest minimal_mcrph/tests`.
That is the point of the change described above: the reference is a program that gets run,
not a file that can silently go stale.

Regenerate it when you want to *look* at a column by hand — it is what
`fortran/README.md`'s worked example refers to, and it is the quickest way to see what one
timestep of the scheme actually does to a profile. Regenerate it again after changing
anything under `fortran/`, or it will describe the previous version of the scheme.

Two things worth knowing if you compare that file against an older copy:

- It is written at `ES24.16E3` (17 significant digits, an exact float64 round-trip). It
  used to be `ES16.8E3` (9 digits), which was itself a ~1e-9 error floor on any comparison
  made against it, so every value will look "changed" against a pre-`ES24.16E3` copy even
  where the physics is identical.
- `satad_only/example/output_fields.csv` and `output_fields_gt4py.csv` *are* tracked,
  because they were committed before those ignore rules existed. Only this variant's have
  to be regenerated.

## References

- Original ICON microphysics: Seifert & Beheng (2006)
- GT4Py: https://github.com/gridtools/gt4py

# minimal_mcrph_gt4py

GT4Py port of the stripped 2-moment microphysics scheme in `../fortran`.

See `../porting_plan/porting_plan.md` for the design doc and `../CLAUDE.md` for the
validation contract and standing decisions. `../port_log.md` tracks progress.

## Fidelity: what the port currently achieves, and what it took

The port reproduces the Fortran to **1.6e-15 relative (~7 ULP)** on every field, at every
process boundary, across all six test columns; the `warm` column is bit-identical. Tests
assert `rtol=1e-14`.

Getting there required fixing three transcription bugs, all of which the previous
`rtol=1e-6` test suite was loose enough to accept. All three are the same *kind* of
mistake — reading a Fortran literal as the number it looks like — and are worth knowing
about before porting anything else from this codebase.

### 1. Fortran default-real literals are single precision

The particle constants in `mo_2mom_mcrph_main.f90` are written without a kind suffix:

```fortran
0.333333, & !..mu.....Exp.-parameter der Verteil.
0.390000, & !..b_geo..Koeff. Geometrie
```

A plain decimal literal in Fortran is *default real* — single precision — and is only
widened to `REAL(wp)` on assignment. So the value the Fortran actually holds for `mu` is
`3.33332985639572144e-01`, not the `3.33332999999999990e-01` you get by typing the same
digits into Python. That is 4.3e-8 relative.

It matters because these constants feed the gamma-function coefficient setup and the
per-level diameter/velocity power laws: every derived coefficient was off by 1e-8 to
4e-7, which moved every depositional-growth increment by ~5e-7 and showed up as a ~2e-7
error in `qi`. `particles.py::_r4` now reproduces the single-precision rounding, with each
field marked according to which form its Fortran literal uses (`d` notation *is* double
and must not be passed through it).

The same trap appears in `ice_nucleation_het_inas`:

```fortran
ddust_background = (/ 0.2_wp, 0.4_wp, 0.6_wp/) * 1e-6
```

— the array elements carry `_wp`, the scale factor does not.

`test_coefficients.py::test_particle_constants_carry_fortran_literal_precision` guards
against someone "simplifying" the `_r4` calls back out again, which would look like a
tidy-up and silently reintroduce the error.

### 2. satad's early exit is part of the answer

`satad_v_3D`'s Newton loop exits when a step moves the temperature by less than
`tol = 1e-3` K. The port originally ran all 10 iterations unconditionally, reasoning that
further steps at a converged fixed point are idempotent. They are *nearly* idempotent —
but the Fortran stops a step short of the fixed point and the port walked all the way to
it, so the two landed on different doubles: 2.6e-13 on temperature, 5.8e-12 on `qv`,
2.1e-11 on `qc`. Since satad runs first, that put a ~1e-11 floor under every field
downstream.

`satad.py` now reproduces the stopping rule. Note *how*: masking each step
(`where(active, step(t), t)`) is the obvious encoding and is ruinously expensive, because
it names the previous iterate four times instead of three and over ten unrolled steps that
is ~18x the expression tree — measured, it took gtfn_cpu from 25 s to over 10 minutes for
this one stencil. The code instead computes the unconditional chain and *selects* the
iterate the Fortran would have stopped at, which is exact (the chains are identical up to
the stop) and costs a flat cascade at the end.

### 3. An out-of-bounds read in the reference

`ccn_activation_sk_4d` decides its gate on `atmo%w(k+1)` with no clamp, while the gradient
test on the same line clamps with `kp1_fl = MIN(k+1, SIZE(atmo%rho))`
(`mo_2mom_mcrph_processes.f90:1725-1728`). `w` is sized `nlev`, so at the lowest level the
Fortran reads one element past the end of the array and the gate is decided by whatever is
in adjacent memory.

The read is guarded behind `cloud%q(k) > nuc_eps`, which is why the bundled column never
trips it — its lowest level is cloud-free. Constructed columns with cloud water at the
surface do: the Fortran activated CCN there off the out-of-bounds value while the port,
which treats "no data past the array end" as a closed gate, did not — a 100% disagreement
in `qnc` with no correct answer to match.

This is a defect in the reference worth reporting upstream. It is not something a port can
reproduce, so the scenarios stay clear of it rather than encoding one accidental outcome;
see `tests/scenarios.py::_dry_bottom_level`.

## How the validation is put together

See `README.md` for how to run it and `agents_plan_test/harness_plan.md` for the design.
The essentials:

- **The reference is the compiled Fortran, run during the test session.** No committed
  CSV, no pasted numbers. The previous suite asserted against 9-12 digit literals captured
  from instrumented runs that were then reverted — unregenerable, and the transcription
  alone capped the achievable tolerance.
- **`fortran/mo_stage_dump.f90`** writes the column at each of the nine process boundaries
  at 17 digits when the driver is given a dump directory, and is a no-op otherwise, so a
  bare `make run` is byte-for-byte unaffected. This is what makes a discrepancy localise
  to a process instead of to the whole timestep.
- **`column_driver.f90`** takes its paths, `dt` and dump directory as optional positional
  arguments (defaults unchanged), and writes `ES24.16E3` rather than `ES16.8E3`. The old
  9-digit output was itself a ~1e-9 error floor on any comparison made against it.
- **Scenario columns are checked against independent thermodynamics**
  (`mcrph_common/thermo.py`, Murphy-Koop) so that a column named `deep_cold` is verified to
  actually be ice-supersaturated. Those functions describe inputs only and must never be
  used to judge a result.

## A few general notes

- **Structure:** Closely aligned with the existing `icon4py` 1-moment port to facilitate full microphysics porting and future ICON integration, though this increases import complexity.
- **Organization:** Sub-processes are currently split into separate files to ease isolated testing, but should be consolidated later.
- **Backend:** Runs with `gtx.gtfn_cpu` (tested with local `gcc`). Note that compilation can take a few minutes.
- **Dependencies:** Uses `icon4py` dependencies for type consistency and requires GT4Py `1.1.11`. Local checkouts (`../../gt4py`, `../../icon4py/model/common`) are configured as editable installs via `[tool.uv.sources]` in `pyproject.toml`.

## Setup

This project is a `[tool.uv.workspace]` member of the repo-root `pyproject.toml`
(`hpc4wc_mcrph_column_gt4py/pyproject.toml`) -- there is **one shared `.venv`
and `uv.lock` at the repo root**, not a separate one per project. Set it up
from the repo root, not from here:

```bash
cd ../..                          # repo root
uv sync --package minimal_mcrph_gt4py --extra dev
```

That installs this package plus its `dev` extras (pytest, pytest-cov) into the
shared root `.venv`, alongside `gt4py`/`icon4py-common` editable from the local
checkouts. Python is pinned via the repo root's `.python-version` (3.13,
matching the environment this was validated on).

Once synced, `uv run ...` works from *either* the repo root or from this
directory -- uv auto-detects the workspace root and always targets the shared
`.venv` either way. All commands below assume you're in this directory.

(The older mamba-based workflow -- `mamba activate hpc4wc` +
`pip install -e ../../icon4py/model/common` + `pip install -e ".[dev]"` -- still
works too if you're not on the shared uv setup, but uv is the supported path
going forward.)

## Tests

See **`README.md`** — how to run the suite, what each layer covers, the measured
tolerances, and how to regenerate `example/output_fields.csv` (which is gitignored, and
which the tests no longer read) all live there, in one place rather than two.

This section previously described the suite as asserting `rtol=1e-6` against a committed
`example/output_fields.csv`. Both of those are gone: the reference is the Fortran binary,
built and run during the test session, and the tolerance is `rtol=1e-14`.

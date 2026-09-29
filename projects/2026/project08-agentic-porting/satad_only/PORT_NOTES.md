# satad_only / gt4py

A **GT4Py** (`gt4py.next` declarative frontend) port of ICON's saturation-adjustment
kernel, independently re-implemented from the Fortran in
[`../fortran/mo_satad.f90`](../fortran/mo_satad.f90) (`satad_v_3D` / `satad_v_3D_gpu`).

## What it does

Saturation adjustment relaxes each grid point to liquid/vapour equilibrium **at constant
total density**, moving water between vapour (`qv`) and cloud water (`qc`) and adjusting
temperature (`T`) by the associated latent heat. Every grid point is independent (no
vertical coupling), so the kernel is purely elementwise.

Per point: if all cloud water can evaporate and the air is still sub-saturated, the state
is computed directly (branch A); otherwise a Newton iteration on temperature is run to
reach saturation (branch B).

Only the active Tetens saturation formula (`ipsat == 1`) is ported. The Murphy–Koop path
(`ipsat == 2`) is dead code in the source and is intentionally not reproduced.

## Files

| File | Purpose |
|---|---|
| `satad_gt4py.py`      | the port: dimensions, constants, field operators, the `satad` program, and a `satad_numpy` wrapper |
| `test_satad_gt4py.py` | standalone self-check (numpy reference comparison + physical + edge-case tests) |

## Public interface

Two entry points, both in `satad_gt4py.py`:

**1. `satad` — the GT4Py `@program`** (operates in place on `(Cell, K)` fields):

```python
import gt4py.next as gtx
from satad_gt4py import satad, CellDim, KDim
# te, qve, qce, rhotot : gtx.Field[Dims[CellDim, KDim], float64]
# tol : float  (temperature tolerance in K, e.g. 1.0e-3)
satad(te, qve, qce, rhotot, tol, offset_provider={})   # te, qve, qce updated in place
```

| Argument | Meaning | Units | Intent |
|---|---|---|---|
| `te`     | temperature          | K       | in/out |
| `qve`    | specific vapour      | kg/kg   | in/out |
| `qce`    | specific cloud water | kg/kg   | in/out |
| `rhotot` | total density        | kg/m³   | in     |
| `tol`    | temperature accuracy | K       | in (scalar) |

**2. `satad_numpy` — a numpy convenience wrapper** (no GT4Py knowledge needed):

```python
from satad_gt4py import satad_numpy
tk_out, qv_out, qc_out = satad_numpy(rho, tk, qv, qc, tol=1.0e-3)
```

Accepts either 1-D arrays `(nlev,)` (a single column) or 2-D arrays `(ncells, nlev)`;
returns adjusted `(tk, qv, qc)` with the same shape. `rho` is unchanged.

### Fixed parameters (documented assumptions)

- **`maxiter` is fixed at `MAXITER = 10`**, matching the reference driver
  `column_driver.f90`. GT4Py forbids data-dependent loops, so the Newton iteration is
  **manually unrolled 10 times**, each step masked so converged / non-iterating points
  freeze. This reproduces the `satad_v_3D_gpu` structure and is numerically identical to
  the `satad_v_3D` `while`-loop. If a different `maxiter` is required, change the number of
  unrolled steps in `_satad`.
- **Working precision is `float64`** throughout (ICON `wp`).
- **Backend defaults to embedded** (pure numpy, `backend=None`) — no C++ toolchain needed.
  `satad_numpy(..., backend=gtx.gtfn_cpu)` selects a compiled backend if available.

## Running

Requires `gt4py` (`gt4py.next`) and `numpy`. With GT4Py installed (or on `PYTHONPATH`):

```bash
python test_satad_gt4py.py
```

The script prints a PASS/FAIL line per test and the per-level change on the bundled
sample column (`../fortran/example/fields.csv`), mirroring what the Fortran driver prints.
It exits non-zero if any test fails. The `test_*` functions are also pytest-discoverable
(`pytest test_satad_gt4py.py`).

## How it was verified

- **Independent numpy reference.** `test_satad_gt4py.py` contains `_satad_reference_numpy`,
  a scalar transcription written directly from the Fortran `while`-loop (not from the DSL
  masking logic). The GT4Py port agrees with it to ~1e-10 on the sample column.
- **Physical checks.** Water (`qv + qc`) is conserved; cloud stays non-negative;
  condensation warms and evaporation cools; a dry sub-saturated point is unchanged; both
  branches are exercised.
- **Layout check.** A 2-D `(ncells, nlev)` block gives the same result as each column run
  on its own.

> Note: agreement with Fortran is expected at ~1e-10, not bit-for-bit, because of
> floating-point operation ordering in `exp` and division. A Fortran compiler was not
> available in the porting environment, so the byte-level reference (`make run` in
> `../fortran`) was not executed here; the numpy transcription stands in for it and can be
> cross-checked against the Fortran output when a compiler is present.

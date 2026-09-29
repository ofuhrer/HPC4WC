# Native Fortran ⇄ GT4Py consistency harness for `satad_only`

## Context

The only executable test for the GT4Py saturation-adjustment port today is
`satad_only/gt4py/test_satad_gt4py.py`. Its core comparison
(`test_matches_numpy_reference`) checks the GT4Py output against
`_satad_reference_numpy` — a **hand-transcribed pure-numpy copy** of the Fortran
`satad_v_3D` while-loop. That transcription exists because no Fortran compiler
was available during the port, so it "stands in" for the real Fortran. A numpy
stand-in can silently share the same porting mistakes it is meant to catch, so
it is not a trustworthy oracle.

This harness establishes consistency by running the **actual compiled Fortran**
and the **actual GT4Py DSL** on identical inputs and comparing their outputs —
no numpy translation anywhere in the trust path. The example CSVs live at
`satad_only/example/` so Fortran, GT4Py, and the harness share one canonical
data location. The harness lives at `satad_only/tests/` and is pytest-only.

Comparison tolerance: `numpy.isclose(rtol=10e-12, atol=0)` by default; `atol!=0`
only for fields computed around zero (i.e. `qc`, which floors at `0.0` /
`ZQWMIN=1e-20`).

## Design decisions

- **Fortran driver gains CLI args + full-precision output** — so the harness can
  drive arbitrary scenario columns through the real binary and read results back
  without the ~8-sig-fig truncation that would otherwise make `rtol=10e-12`
  impossible.
- **Test inputs = bundled `example/fields.csv` column + generated edge cases**
  (dry/unchanged, evaporation branch A, supersaturation branch B, near-zero qc).
- **Runner = pytest only**, fully self-contained (stdlib `tempfile` for the
  driver's I/O scratch files; no dependence on any external scratch dir).

## Known risk to surface, not hide

The port docs *estimate* GT4Py-vs-Fortran agreement at ~1e-10 (never measured;
no compiler was available). `rtol=10e-12` (=1e-11) is the target. GT4Py's
embedded backend uses numpy's `exp`/division while gfortran uses its libm —
these can differ at the ULP level and, through the Newton iteration, may exceed
1e-11 for `tk`. The harness **attempts 10e-12 as specified**; if `tk` fails
solely on libm divergence, that is a finding to report (with the observed max
deviation), not a reason to quietly relax the bound.

---

## Step 2 — Make the Fortran driver harness-callable

Edit `satad_only/fortran/column_driver.f90`:

- **Optional positional CLI args**, all defaulted so existing behaviour is
  preserved: `column_driver [input_csv [output_csv [tol [maxiter]]]]`.
  Parse with `COMMAND_ARGUMENT_COUNT()` / `GET_COMMAND_ARGUMENT(...)`; convert
  `tol` and `maxiter` via internal `READ`. The `fields_file` / `output_file` /
  `tol` / `maxiter` `PARAMETER`s become variables initialised to the defaults.
- **Fix default paths** to the new location: `../example/fields.csv` /
  `../example/output_fields.csv` so a bare `make run` from `fortran/` works.
- **Full-precision output**: change the `write_fields` format from `ES16.8E3`
  (~8 sig figs) to `ES24.16E3` (~17 sig figs) so a round-trip preserves float64.
  Required to compare at `rtol=10e-12`.

No change to `mo_satad.f90` or the Makefile logic (`make driver` already builds
`build/column_driver`; `.NOTPARALLEL` stays — no `make -j`).

## Step 3 — Fortran runner helper

`satad_only/tests/fortran_runner.py` (file/subprocess plumbing only, no numpy
translation):

- `ensure_driver_built()` → `subprocess.run(["make", "driver"], cwd=<fortran>)`
  once per session. Honours an optional `FC` override via env.
- `run_fortran(rho, tk, qv, qc, tol, maxiter) -> (tk_out, qv_out, qc_out)`:
  write the input column to a temp CSV, invoke
  `build/column_driver <tmp_in> <tmp_out> <tol> <maxiter>`, read `<tmp_out>`
  with `np.loadtxt(delimiter=",", skiprows=1)`, return columns `[:,1:4]`.
  Non-zero exit → clear failure.
- **Both temp files come from stdlib `tempfile`** (OS temp dir), one set per
  call, removed after reading. The driver's job is to *write an output CSV*; if
  the harness aimed it at the committed `example/output_fields.csv`, every run
  would leave that version-controlled file dirty and overwrite it with whichever
  scenario ran last. Temp files keep the committed golden file untouched — it is
  only rewritten by the deliberate regeneration in Step 7.

## Step 4 — GT4Py bridge + shared fixtures

`satad_only/tests/conftest.py`:

- Put `satad_only/gt4py/` on `sys.path`, import `satad_numpy`, `DEFAULT_TOL`,
  `MAXITER`, `ZQWMIN` from `satad_gt4py.py`. GT4Py runs on the default embedded
  (numpy-DSL) backend — real DSL execution, not a translation. (`gtfn_cpu` is
  skipped: per `performance_backend_notes.md` the backend arg only sets the
  allocator, so it would not actually run compiled.)
- Session fixture calling `ensure_driver_built()`.
- `compare(name, got, ref, atol=0.0)` wrapping
  `np.isclose(got, ref, rtol=10e-12, atol=atol)`, reporting offending indices and
  max abs/rel deviation on failure. Default `atol=0`; `qc` passes a small `atol`
  (`1e-18`, just above `ZQWMIN`).
- `tol`/`maxiter` are passed identically to both sides (`1e-3`, `10`) so the two
  solve the same problem.

## Step 5 — Scenario definitions

`satad_only/tests/scenarios.py`, each returning `(name, rho, tk, qv, qc)`:

- `bundled`: load `satad_only/example/fields.csv`.
- `dry_unchanged`: warm, very dry, no cloud → passes through.
- `evaporation_A`: cloud in sub-saturated air → branch A cools, cloud evaporates.
- `supersaturation_B`: cold, vapour-rich, some cloud → branch B warms/condenses.
- `near_zero_qc`: mix producing `qc` at/around `0.0` and `ZQWMIN` — needs `atol`.
- `mixed_random`: a fixed-seed multi-level column spanning both branches.

## Step 6 — The consistency test module

`satad_only/tests/test_satad_consistency.py`:

- `@pytest.mark.parametrize` over scenarios: run `run_fortran(...)` and
  `satad_numpy(...)` on identical inputs, then `compare` `tk`, `qv`,
  `qc` (atol=1e-18), and assert `rho` passes through unchanged.
- A physical-invariant test (no oracle needed): `qv+qc` conserved vs input;
  condensation warms / evaporation cools — on the GT4Py output.

## Step 7 — Retire the numpy-translation test

Delete `satad_only/gt4py/test_satad_gt4py.py` (its `_satad_reference_numpy`
oracle and broken `../fortran/example/` path are what this rewrite removes).
Regenerate `satad_only/example/output_fields.csv` at full precision via the
updated driver.

---

## Files

- **Modify**: `satad_only/fortran/column_driver.f90`.
- **New**: `satad_only/tests/{conftest.py, fortran_runner.py, scenarios.py,
  test_satad_consistency.py}`.
- **Delete**: `satad_only/gt4py/test_satad_gt4py.py`.
- **Regenerate**: `satad_only/example/output_fields.csv` (full precision).

## Verification

1. `make -C satad_only/fortran driver` builds; `cd satad_only && ./fortran/build/column_driver`
   (no args) reproduces the reference run against `example/fields.csv`.
2. `./fortran/build/column_driver <in.csv> <out.csv> 1.0e-3 10` writes a
   full-precision `out.csv` that round-trips to float64.
3. `cd satad_only && python -m pytest tests/ -v` — all scenarios pass at
   `rtol=10e-12` (with `atol` only on `qc`).
4. If any `tk` comparison fails only by a small libm-`exp` margin, report the
   observed max deviation and confirm whether to keep 10e-12 or relax `tk`
   specifically — do **not** loosen silently.

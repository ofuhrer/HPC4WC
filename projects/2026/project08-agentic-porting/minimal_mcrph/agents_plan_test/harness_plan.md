# Rebuild `minimal_mcrph`'s validation harness on live Fortran + `compare()`

## Context

The GT4Py port in `minimal_mcrph/` is, structurally, a careful piece of work: the process
order matches `clouds_twomoment`, the density/mixing-ratio conversions and the easily-missed
post-`post_twomoment` background relaxation (`mo_2mom_mcrph_driver.f90:467-524`) are all
there, and the tricky Fortran quirks are documented rather than papered over (the
`Xi_i < eps` signed comparison, `dep_sum`'s ice/graupel/snow/hail summation order, the
`x_i` snapshot taken before the `q` update, `R_d => rv` being the *vapour* gas constant).
Every constant I cross-checked against the Fortran — `K_T`/`con0_h`, `nu_l`/`con_m`,
`N_sc`, `n_f`, `L_ed`/`als`, `cp_v`, `clw`, the Tetens `b1/b2i/b3/b4i` aliases, and the
`vent_coeff_a`/`vent_coeff_b`/`moment_gamma` formulas — is transcribed correctly.

**The tests around it are the weak part, and they are weak in a way that hides a real
discrepancy.** Three compounding problems:

1. **The reference file was the error floor.** `column_driver.f90` wrote
   `example/output_fields.csv` with `ES16.8E3` — 9 significant digits. No comparison against
   that file could ever resolve better than ~1e-9 relative, so "agrees to 1e-9" was measuring
   the file format, not the port. (`satad_only`'s driver already writes `ES24.16E3`.)
2. **Nothing runs the Fortran.** Every per-process unit test asserts against hardcoded
   9–12 digit literals transcribed from instrumented Fortran runs that were then reverted.
   They cannot be regenerated, and the transcription caps achievable `rtol` at ~1e-12. This is
   the opposite of `satad_only`, which builds and runs the real binary in-process. The root
   cause is mechanical: `minimal_mcrph/fortran/column_driver.f90` hardcoded its I/O paths, so
   `mcrph_common.fortran_runner.run_driver` could not drive it.
3. **Tolerances are set by what the literals support, not by what the port achieves.**
   `rtol=1e-6` in the integration test, `1e-6`–`1e-9` in the unit tests, `np.testing.assert_allclose`
   everywhere, and `mcrph_common.compare()` used nowhere. `atol=1e-20` is applied uniformly —
   meaningless next to `qnc ~ 1e8`, too tight next to `qc` floored at `ZQWMIN`.

I already fixed (1) — added CLI args and `ES24.16E3` to the driver, regenerated the reference.
Measured error dropped ~100x immediately (`tk` 1.27e-9 → 1.76e-11, `nccn`/`ninpot` → exactly 0).
**But `qi` did not follow: it still differs by ~2e-7 relative (5e-13 absolute) at levels 3-8**,
with `qv`/`tk`/`qc` carrying the knock-on downstream through the final satad. The explanation
written into `test_vapor_deposition.py`'s docstring ("longer chain of nested `exp(log(...))`…
amplifies the apparent relative error without indicating an actual bug") does not survive
checking: I measured the `s_si` cancellation amplification at only 2.6–85x, upstream `qv`
agrees to 2e-11, and all the deposition constants match. **A ~5e-7 relative error in the
depositional-growth increment is real and unexplained** — and `rtol=1e-6` was set just loose
enough to accept it.

Two coverage holes make it worse: the bundled column has `w == 0` at every level (so CCN
activation's active branch and KHL06 homogeneous nucleation are never exercised against the
Fortran) and `qs = qns = qg = qng = qh = qnh = 0` at every level (so three of the four
deposition species are dead code in every test).

**Outcome:** a `minimal_mcrph/tests/` harness built the way `satad_only`'s is — real Fortran
binary as the only oracle, `compare()` with per-field tolerances justified by measurement,
scenario columns that reach the untested branches — plus a root-caused answer for the `qi`
residual.

---

## Part A — Fortran driver I/O (already applied, keep)

`minimal_mcrph/fortran/column_driver.f90`:
- `parse_cli` added (same shape as `satad_only/fortran/column_driver.f90:105-133`): positional
  args 1-4 = fields CSV, output CSV, hhl CSV, `dt`; each falls back to its old hardcoded default,
  so bare `make run` is unchanged.
- Output format `ES16.8E3` → `ES24.16E3` (17 digits, exact float64 round-trip).
- `example/output_fields.csv` regenerated.

No physics touched. Verify with `cd minimal_mcrph/fortran && make run` and confirm the file
round-trips.

## Part B — Full-precision stage dumps in the Fortran

Goal: make every mid-pipeline reference value reproducible, so no test ever again asserts
against a transcribed literal.

Add `minimal_mcrph/fortran/mo_stage_dump.f90` (new module, listed in the `Makefile`'s `SRCS`
after `mo_2mom_mcrph_types.f90`):
- module state `stage_dump_dir` (empty = disabled) plus `set_stage_dump_dir`.
- `dump_stage(name, ...)` writes `<dir>/<name>.csv` at `ES24.16E3` with a fixed, name-keyed
  header available at every call site:
  `rho,pres,w,tk,qv,qc,qnc,qr,qnr,qi,qni,qs,qns,qg,qng,qh,qnh,nccn,ninpot,ninact`.
  Column *set* need not match the driver's 23 — `mcrph_common.csv_io.read_columns` is keyed by
  name, not position.
- No-op returns immediately when `stage_dump_dir` is empty, so default runs are byte-identical.

Insert `CALL dump_stage(...)` at the nine boundaries the port mirrors:

| stage | Fortran site |
|---|---|
| `satad_pre` | `mo_nwp_gscp_interface.f90`, after the first `satad_v_3d` (~line 130) |
| `prepare` | `mo_2mom_mcrph_driver.f90:379`, after `prepare_twomoment` |
| `ccn` | `mo_2mom_mcrph_main.f90:575`, after `ccn_activation_sk_4d` |
| `default_n` | `mo_2mom_mcrph_main.f90:584`, after `set_default_n` |
| `ice_nuc` | `mo_2mom_mcrph_main.f90:596`, after `ice_nucleation_homhet` |
| `cloud_freeze` | `mo_2mom_mcrph_main.f90:600`, after `cloud_freeze` |
| `vapor_dep` | `mo_2mom_mcrph_main.f90:613`, after `vapor_dep_relaxation` |
| `ice_melt` | `mo_2mom_mcrph_main.f90:621`, after `ice_melting` |
| `post` | `mo_2mom_mcrph_driver.f90:462`, after `post_twomoment` |

`column_driver.f90` gains arg 5 = dump directory. One Fortran run then yields every stage's
reference at full precision.

Also dump `setup_particle_coeffs`' `a_f/b_f/c_i/c_z` and `init_2mom_sedi_vel`'s coefficients
to `<dir>/coeffs.csv` from `init_2mom_scheme_once` — this replaces `test_coefficients.py`'s
non-regenerable literals and is needed for Part E hypothesis 2.

These are pure-output insertions into reference physics files. They are guarded and default-off,
but they *are* edits to the reference — call that out in `PORT_NOTES.md`.

## Part C — Python harness, mirroring `satad_only/tests/`

Reuse, do not reimplement: `mcrph_common.compare.compare`,
`mcrph_common.fortran_runner.{ensure_driver_built, run_driver}`,
`mcrph_common.csv_io.{read_columns, write_columns}` (already `%.17e`),
`minimal_mcrph.csv_io.read_fields_csv`, `minimal_mcrph.driver.{Driver, ColumnState}`.

**`minimal_mcrph/tests/fortran_runner.py`** — models
`satad_only/tests/fortran_runner.py`. Adds only this variant's column layout (the 23-name
tuple already in `minimal_mcrph/csv_io.py::_COLUMNS`) and its extra inputs. Note
`run_driver` writes exactly one input CSV; `hhl` is a second file, so this module opens its
own `tempfile.TemporaryDirectory`, writes `hhl.csv` there, and passes that path plus `dt`
(and optionally the dump dir) through `extra_args`. No change to `mcrph_common` needed.
Returns the final column as `{name: array}` and, when a dump dir was requested, the per-stage
columns too.

**`minimal_mcrph/tests/conftest.py`** — models `satad_only/tests/conftest.py`:
session-scoped autouse `ensure_driver_built()`; a `compare()` wrapper carrying this variant's
default `rtol`; a per-field tolerance table. Start at `RTOL = 1e-12`, then pin each field to
the measured maximum with a comment recording that measurement — tolerances are asserted
numbers, not aspirations. Fields needing an `atol` floor and why:
`qc` (floors at `ZQWMIN = 1e-20` / exact `0.0`), the frozen-species `q`/`n` pairs (clipped to
exact `0.0`), `ninact`/`ninpot` (relaxed toward a background profile, land on exact values).
Number concentrations ~1e8 get `atol=0` and pure `rtol`.

**`minimal_mcrph/tests/scenarios.py`** — models `satad_only/tests/scenarios.py`. Each scenario
is a named `(fields dict, hhl array, dt)`. Beyond `bundled` (today's only case):
- `updraft` — bundled column with `w > 0` at several levels. Opens the CCN activation gate
  *and* the KHL06 homogeneous-nucleation branch, neither of which any live-Fortran test
  currently reaches.
- `mixed_species` — nonzero `qs/qns/qg/qng/qh/qnh`. Today three of four deposition species are
  never exercised.
- `warm` — `T > 273.15` throughout, no ice. Isolates satad + `ice_melting`; deposition inert.
- `deep_cold` — strongly ice-supersaturated with `qi/qni`, exercising fast deposition.
- `mixed_random` — fixed-seed multi-level column for breadth (same idea as
  `satad_only/tests/scenarios.py::_mixed_random`).

Scenario columns must keep `hhl` physical: `nlev+1` rows, monotonically decreasing (the
post-housekeeping relaxation reads `zf` from it, and `dz = hhl(k) - hhl(k+1)` must stay positive).

### Scenario columns must be verified against thermodynamics, not just asserted

A scenario named `deep_cold` is worthless if the numbers I picked turn out to be
*sub*saturated over ice, and `warm` is worthless if a level quietly sits above water
saturation — the test would still pass and still cover nothing. So every scenario's stated
physical intent gets checked against independent thermodynamics.

**New file `mcrph_common/thermo.py`.** The functions come from
`/Users/bbuchenau/cloud_code/Supersaturation/parcelmodel/utils.py`, but that checkout lives
outside this repo and can't go into `pyproject.toml` without breaking `uv sync` elsewhere — so
hoist the handful actually needed into this repo, with a module docstring naming the source
checkout, the Murphy–Koop / Lohmann-and-Mahrt provenance of each fit, and its validity range.
A collaborator opening the file should be able to see immediately where the maths came from
and that it is *not* the scheme's own thermodynamics. Hoist:

- `Ew(T)`, `Ei(T)` — saturation vapour pressure over water / ice (Murphy–Koop)
- `e(qv, p)` — vapour pressure from mixing ratio
- `S(e, Ew)`, `Si(e, Ei)` — saturation **ratio** (1.0 = saturated). Rename or re-document on
  the way in: the originals are called "supersaturation" but return `e/Ew`, so "slightly
  subsaturated" is `S ≈ 0.98`, not `≈ -0.02`. Getting this backwards silently inverts a
  scenario's meaning.
- `q_v(S, p, T)` — invert a target saturation ratio back to `qv`, so columns get *constructed*
  from the regime they're meant to be in rather than hand-tuned and checked afterwards
- `rho_vs(T)`, `rho_vis(T)` — vapour density at water / ice saturation, directly comparable to
  the density-space state after `prepare_twomoment` and to the scheme's own
  `s_si = qv·R_v·T/e_es − 1`
- the constants they need (`Rv`, `Ra`, `M_w`, `M_a`, `T_0`) from `parcelmodel/constants.py`

Note in the docstring that these constants deliberately differ slightly from the scheme's
(`Rv = 461.5` here vs ICON's `461.51`, `Ra = 287.1` vs `287.04`) — they belong to the
independent reference, and must not be "corrected" to match `minimal_mcrph/constants.py`.

Why this is a real check and not circular: these are Murphy–Koop fits, while the scheme under
test uses Tetens (`b1·exp(b2i·(T−b3)/(T−b4i))`). The two agree to ~1% on vapour pressure —
close enough to confirm "this level is ice-supersaturated by roughly 20%", nowhere near close
enough to serve as a numerical oracle. **These characterise and construct scenario inputs;
they never validate port outputs** — the Fortran binary stays the only oracle for results.
`mcrph_common/thermo.py` needs a header saying exactly that, because it is the one file in the
tree where someone could plausibly mistake a second implementation of the physics for a
reference.

Then `minimal_mcrph/tests/test_scenario_sanity.py`: for each scenario, assert the regime it
claims — `warm` has `S < 1` and `T > T_3` at every level; `deep_cold` has `Si > 1` wherever
`qi > 0`; `updraft` has `w > 0` and is at or above water saturation at the levels where the
CCN gate is expected to open; near-zero-`qc` levels really do sit just above saturation.
Build the `qv` values with `q_v(S, p, T)` from a target `S`, and record the resulting `S`/`Si`
per level as comments in `scenarios.py` so the intent is legible without running anything.

Placing this in `mcrph_common/` rather than `minimal_mcrph/tests/` follows the repo's own rule
that shared, variant-agnostic helpers live there — `satad_only`'s `near_zero_qc` scenario had
to hand-tune `qv` values and verify them after the fact, which is exactly the job `q_v()` does
properly, so that variant can adopt it later.

## Part D — Rewrite the tests on `compare()`

**`tests/integration_tests/test_full_column.py`** — parametrize over every scenario × both
backends; run the Fortran live via `run_fortran(...)` instead of reading the committed CSV;
assert with `compare()` and the per-field tolerances. Keep the committed
`example/output_fields.csv` as a documented `make run` artifact, no longer as the test oracle.

Add `test_physical_invariants` (oracle-free, modelled on
`satad_only/tests/test_satad_consistency.py:41-54`): total water conserved across the step
within the deposition/nucleation budget, all `q`/`n` non-negative, mean particle mass `q/n`
inside `[x_min, x_max]` for every species after the final clipping.

**`tests/unit_tests/`** — add a stage-boundary test that drives `Driver.run_timestep` with a
new optional `stages` recorder (a small addition to `minimal_mcrph/driver.py`: when passed a
dict, it stores a copy of the column at the same nine boundaries) and `compare()`s each stage
against the Fortran dump. This subsumes what the literal-based per-process tests were reaching
for, at full precision and reproducibly.

Then, per module:
- `test_satad.py`, `test_ice_nucleation.py`, `test_processes.py`, `test_vapor_deposition.py`,
  `test_ccn_activation.py` (`_noop`/`_active_branch`), `test_unit_conversion.py`'s two
  Fortran-referenced tests: replace the literal blocks with values read from the stage dumps,
  and route assertions through `compare()`.
- `test_coefficients.py`: compare against `coeffs.csv` from Part B.
- `test_housekeeping.py`, `test_unit_conversion.py::test_clip_negative` /
  `test_convert_fields_round_trip`: these are hand-derived/self-consistency checks, not
  Fortran-referenced. Keep them, but say so plainly in the docstrings rather than implying
  Fortran provenance. Fix `assert_allclose(cq_out[1], 0.0)` in `test_ccn_activation.py:117`,
  which is an accidental exact-equality check (`desired=0` makes `rtol` inert).
- Correct the two docstrings that overclaim: `test_ccn_activation.py`'s "All 5 matched
  exactly" (the test checks `rtol=1e-6` against 5-significant-digit values) and
  `test_vapor_deposition.py`'s exp/log-amplification explanation (disproved — see Part E).

## Part E — Root-cause the `qi` residual

The stage dumps make this a clean bisection: feed the port's `vapor_dep_relaxation` the
full-precision `cloud_freeze` stage output and compare against the `vapor_dep` stage. If
~5e-7 persists with bit-identical inputs, the error is inside the stencil. Hypotheses in
priority order:

1. **The `f_v` limiter branch.** For the bundled column's ice I computed
   `f_v = a_f + b_f·sqrt(D·v) ≈ 0.718` against `a_f/a_ven ≈ 0.755` — so
   `MAX(f_v, a_f/a_ven)` *binds*, and the result depends only on `a_f/a_ven`. A branch or
   `a_f` discrepancy here would produce exactly this kind of small systematic offset, and would
   also explain why the error is insensitive to the `b_f`/`sqrt(D·v)` chain the current
   docstring blames.
2. **`a_f`/`b_f` at full precision.** `test_coefficients.py` checks `rtol=1e-6` against
   9-digit literals — a 5e-7 error passes today. Compare against `coeffs.csv` at `1e-14`.
3. **`dep_rate_ice`/`dep_rate_snow`.** The Fortran accumulates into these
   (`mo_2mom_mcrph_processes.f90:1454-1455`); confirm nothing in the stripped scheme consumes
   them, i.e. that the port is right to drop them.
4. **`math.gamma` vs gfortran `GAMMA`** for the ICE argument ~5.079 — expected ~1 ULP, so this
   is a floor check, not a likely cause.

Whatever the answer, record it in `PORT_NOTES.md` and set the final tolerance from it. If it
turns out to be a port bug, fix the stencil; if it is genuinely benign, the note must say
*why*, with the measurement.

## Part F — Docs and plan record

Create `minimal_mcrph/agents_plan_test/` and save this plan there as `harness_plan.md`,
matching `satad_only/agents_plan_test/harness_plan.md` and the repo README's rule that agent
plans get copied into the folder they apply to (`minimal_mcrph/agents_plan_port/` already
holds the porting-side equivalents).

`minimal_mcrph/README.md` and `PORT_NOTES.md` currently advertise `rtol=1e-6` and the
committed-CSV comparison as "the actual definition of 'the port is correct'". Update to
describe the live-Fortran harness, the measured tolerances, the new driver args and stage-dump
flag, and the Part E finding. Refresh `scripts/compare_to_reference.py` to run the Fortran
live and print the same per-field table.

---

## Verification

```bash
# Fortran side unchanged by default, and reproducible
cd minimal_mcrph/fortran && make driver && make run
git diff --stat minimal_mcrph/example/output_fields.csv   # only the precision bump

# stage dumps land where expected
./build/column_driver ../example/fields.csv /tmp/out.csv ../example/hhl.csv 30.0 /tmp/stages
ls /tmp/stages   # satad_pre ccn default_n ice_nuc cloud_freeze vapor_dep ice_melt post coeffs

# harness
cd ../.. && uv run pytest minimal_mcrph/tests -v        # ~7 min, gtfn_cpu compile dominated
uv run pytest minimal_mcrph/tests -k embedded -v        # fast loop, no compile
uv run pytest minimal_mcrph/tests/test_scenario_sanity.py -v   # scenarios are the regime they claim
uv run pytest                                            # satad_only must stay green

uv run python minimal_mcrph/scripts/compare_to_reference.py
```

Done when: every assertion routes through `compare()`; no hardcoded Fortran-derived literal
remains in `minimal_mcrph/tests/`; every tolerance in `conftest.py` carries a comment stating
the measurement that justifies it; every scenario's claimed regime is confirmed against
`mcrph_common/thermo.py` rather than asserted in a docstring; the `updraft` and `mixed_species`
scenarios pass (proving the CCN-active, homogeneous-nucleation and snow/graupel/hail
deposition paths are actually covered); and the `qi` residual is either fixed or explained in
writing with numbers.

---

# Outcome (recorded after implementation)

This plan is kept as written, including the parts that turned out to be wrong. The
hypotheses in Part E in particular were mostly wrong, and how they were wrong is more
useful to a future reader than a tidied-up version would be.

## What the `qi` residual actually was

Part E guessed at the `f_v` limiter branch, at gamma-function differences, and at the
`dep_rate_ice` accumulators. None of those was the cause.

The cause was that **Fortran plain decimal literals are default real, i.e. single
precision**. The particle constants in `mo_2mom_mcrph_main.f90` are written `0.333333`,
not `0.333333_wp`, so the Fortran holds `3.33332985639572144e-01` where the port, having
transcribed the same digits into Python, held `3.33332999999999990e-01`. Every
gamma-derived coefficient was off by 1e-8 to 4e-7 — which `test_coefficients.py`'s
`rel=1e-6` could not see.

Two further bugs of the same family fell out of the stage-by-stage bisection the plan set
up, neither of them anticipated:

2. **satad's early exit is part of the answer.** Running all 10 Newton iterations
   unconditionally is not equivalent to stopping at `tol`; it cost 2.1e-11 on `qc` and,
   because satad runs first, floored every field downstream.
3. **`ddust_background`'s `1e-6` scale factor** in `ice_nucleation_het_inas` carries no
   `_wp` either, while the array it multiplies does.

Result: worst-case agreement went from 3.2e-7 to 1.6e-15 (~7 ULP), and `rtol` from 1e-6 to
1e-14. Full detail in `../PORT_NOTES.md`.

## Where the plan was right

The stage-dump mechanism (Part B) was what made all three findings tractable — each one
localised to a single boundary in one run. Fixing the reference file's precision (Part A)
was a precondition for seeing any of it: at 9 significant digits the whole investigation
would have bottomed out at ~1e-9.

## Deviations from the plan

* **Per-process unit tests were deleted rather than re-based.** The plan said to replace
  their pasted literals with stage-dump values. Once `test_stage_boundaries.py` existed
  that would have been duplicate coverage of the same processes, at coarser granularity
  and with more machinery, so `test_satad.py`, `test_ice_nucleation.py`,
  `test_processes.py` and `test_vapor_deposition.py` are gone. What remains in
  `unit_tests/` is what the stage comparison genuinely cannot reach: the coefficient
  setup, the CCN table away from the sampled cells, and closed-form properties.
* **Scenario columns needed a constraint the plan did not foresee**: no cloud water on the
  lowest level, because the reference reads `atmo%w(k+1)` out of bounds there. Three of
  the six scenarios initially tripped it.
* **The naive encoding of satad's stopping rule was unusable.** Masking each step took
  gtfn_cpu compilation from 25 s to over 10 minutes for that one stencil; the selection
  form in `satad.py` is exact at 58 s.

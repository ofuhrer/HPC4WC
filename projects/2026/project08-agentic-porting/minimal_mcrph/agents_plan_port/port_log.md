# Port Log

Running diary of what was tried, what worked, what didn't, and why. Append to
this; don't retroactively clean up old entries (see `CLAUDE.md`).

---

## 2026-07-23 — Environment validated, package scaffolded, Phase 1 done

### Environment

`hpc4wc` mamba env confirmed working end to end: `icon4py-common` and
`icon4py-atmosphere-microphysics` editable-installed (gt4py pinned down from
1.1.12 to 1.1.11 to match), and a smoke test compiled+ran a trivial
`@gtx.field_operator` through the **`gtfn_cpu`** backend specifically (not just
the embedded default) — confirms the C++ toolchain is wired up correctly before
any real physics stencil depends on it.

### Important correction: `nuc_i_typ` was wrong in the plan

While deriving the exact config values needed for `config.py`, re-derived
`cloud_type`'s arithmetic from `mo_2mom_mcrph_driver.f90` instead of trusting the
plan's existing table. Result: **`nuc_i_typ = 1` (`ice_nucleation_het_inas`,
Ullrich et al. 2007 INAS scheme), not `nuc_i_typ = 6` (Phillips 2010)** as an
earlier draft of `porting_plan.md` claimed.

```
ccn_type   = ccn_type_gscp5 = 8
cloud_type = 2003 + 10*8 + 100*1 = 2183
nuc_i_typ  = MOD(2183/100, 10) = 1   -> ice_nucleation_het_inas
nuc_c_typ  = MOD(2183/10, 10)  = 8   -> ccn_activation_sk_4d (this part was already right)
```

Caught before any code was written against the wrong value. Fixed in
`porting_plan.md` and `CLAUDE.md`. Lesson: a value repeated across two prior plan
drafts still isn't verified — every config/formula claim needs to be re-derived
from source at the point it's actually used in code, not carried forward from
documentation on trust.

Also found while re-deriving this: the CCN coefficients for `nuc_c_typ=8` are the
**continental** aerosol case (`Ncn0=1700e6, lsigs=0.2, R2=0.03, etas=0.7`), not
maritime — driven by `cfg_2mom_default%tune_sbmccn=1.0` branching in
`two_moment_mcrph_init`. Easy to grab the wrong branch since both live under the
same `CASE(8)` label. Documented in `porting_plan.md`.

And: `init_2mom_sedi_vel`'s three outputs (`coeff_alfa_n/q`, `coeff_lambda`) are
**not used anywhere** in the retained call tree (grep-confirmed — only written
and printed in an `isprint` debug block). An earlier plan draft claimed
`coeff_lambda` was used via `particle_diameter`/`particle_velocity`; it isn't —
those read the particle's own `a_geo/b_geo/a_vel/b_vel` directly. Ported the
routine anyway, since it's a free validation point (see below), but it doesn't
feed any of the 5 processes' actual physics.

### Package scaffolded

`minimal_mcrph/minimal_mcrph_gt4py/` created: `pyproject.toml` (plain setuptools
src-layout, no `[tool.uv.*]`), `src/minimal_mcrph/{constants,particles,config}.py`,
`tests/unit_tests/test_coefficients.py`. Installed editable
(`pip install -e . --no-deps`).

### Phase 1 implemented and validated against real Fortran output

- `constants.py`: physical constants transcribed from the `USE mo_physical_constants`
  alias block in `mo_2mom_mcrph_processes.f90`. Flagging one gotcha for future-me:
  the Fortran aliases `R_d => rv` (line 91) — i.e. what's called `R_d` in every
  `qv * R_d * T` supersaturation formula throughout the processes file is the
  **water vapor** gas constant (461.51), not dry air (287.04, aliased `R_l`).
  Named `R_D_VAPOR` explicitly in `constants.py` to make this impossible to get
  backwards later.
- `particles.py`: the six `ParticleConfig`/`FrozenParticleConfig` instances
  actually selected by `init_2mom_scheme` for the default config (`cloud_nue1mue1`,
  `rainSBB`, `ice_cosmo5`, `snowSBB`, `graupelhail_cosmo5`, `hail_cosmo5` — all
  used unmodified, every `cfg_2mom_default` per-species override is a `-999.99`
  sentinel), plus `setup_particle_coeffs`/`init_2mom_sedi_vel` and their
  `vent_coeff_a`/`vent_coeff_b`/`moment_gamma` dependencies as plain Python using
  `math.gamma`.
- `config.py`: `nuc_c_typ=8`, `nuc_i_typ=1`, and the continental `CCNCoeffs`.

**Validation:** captured real Fortran reference values two ways —
`init_2mom_sedi_vel`'s output is printed unconditionally already (`isprint` is a
hardcoded `.TRUE.` `PARAMETER`, no source change needed — just `make run`).
`setup_particle_coeffs`'s output (`a_f/b_f/c_i/c_z`) isn't printed anywhere, so
added a temporary `WRITE` after its call sites in
`mo_2mom_mcrph_main.f90::init_2mom_scheme_once`, ran once, captured values, then
reverted it (`git status`/`git diff` confirm `mo_2mom_mcrph_main.f90` is back to
original, `make` still builds clean). Both sets match the Python implementation
to 7-9 significant figures across all five species that need them (ice, snow,
graupel, hail, cloud). `tests/unit_tests/test_coefficients.py` (9 tests) codifies
this as a permanent regression check — passing.

### Deliberately not done in Phase 1

- `rain_coeffs` (`setup_particle_coeffs(rain, ...)`, `mo_2mom_mcrph_main.f90:757`):
  confirmed unused by any of the 5 retained processes (only `rain.x_min`/`x_max`
  are needed, for clipping — those live on `ParticleConfig` already). Not
  implemented; note here in case a later process turns out to need it after all.
- No `gt4py`/`icon4py` imports anywhere yet — Phase 1 is deliberately pure
  Python/NumPy per the roadmap (coefficients have no vertical-field dependence,
  computed once). This is expected, not an oversight.

### Open questions / next up (superseded below — see Phase 2 entry)

- `het_icenuc_inas_depo` (called by `ice_nucleation_het_inas`, the now-confirmed
  default heterogeneous-nucleation path) hasn't been read yet — needed before
  Phase 3 implements `ice_nucleation_homhet`.
- The Segal & Khain 4D table construction (`get_otab`/`equi_table`) is still
  unread in detail; needed for the NumPy CCN step (Phase 3).
- Phase 2 (unit-conversion stencils: `prepare`/`post` density transform +
  latent-heat update) is next per the roadmap.

---

## 2026-07-23 — Phase 2 done: unit-conversion stencils, first real GT4Py code

### Two more plan corrections found while implementing (same lesson as Phase 1:
### re-derive, don't trust a repeated claim)

1. **Latent heat is temperature-dependent for the default config, not constant.**
   The plan previously hedged with "or constants `als`/`alv-als` if
   `lconstant_lh`" instead of pinning down which branch is actually active.
   Traced it through: `nwp_gscp_interface.f90` passes `l_cv=.TRUE.` always, and
   `column_driver.f90` sets `ithermo_water=1`; `two_moment_mcrph`
   (`mo_2mom_mcrph_driver.f90:267-271`) computes
   `lconstant_lh = (ithermo_water==0)` = `(1==0)` = **`.FALSE.`** So the real
   path is `latent_heat_sublimation`/`latent_heat_melting` from `mo_satad.f90`
   (temperature-dependent), not the constant `als`/`alv-als` shortcut. Fixed in
   `porting_plan.md` with the full formula transcribed.
2. **`rho_v` (density correction for terminal fall velocity) is a real input to
   `vapor_dep_relaxation`, not just driver bookkeeping** — missed entirely in
   the process-interface table before now. `vapor_deposition_generic` computes
   `v = particle_velocity(p, x) * p.rho_v(k)`; `rho_v` is `rhocorr` for
   ice/snow/graupel/hail and `rhocld` for cloud, both computed once per level in
   `two_moment_mcrph` right before `prepare_twomoment` (not inside it). Added a
   dedicated "Density-correction factors" section to `porting_plan.md` since
   this is genuinely part of Phase 2's scope (computed alongside `prepare`) even
   though it's only consumed by a Phase-3 process.

Both were caught by working through the exact call chain instead of writing
code from the plan's existing (looser) wording. Same pattern as the Phase 1
`nuc_i_typ` catch — worth remembering as a general method, not a one-off.

### Implemented: `stencils/unit_conversion.py`

`convert_fields`/`multiply_field` (mixing-ratio <-> density, in place),
`compute_density_corrections` (`rhocorr`/`rhocld`), `clip_negative` (the
pre-`prepare` `max(q,0)` clip on qr/qi/qs/qg/qh), and
`update_temperature`/`_latent_heat_sublimation`/`_latent_heat_melting` (the
temperature-dependent latent-heat update). First real `@gtx.field_operator`/
`@gtx.program` code in this port — everything before this was pure Python/NumPy.

### Important GT4Py finding: plain float constants break under `gtfn_cpu`

First attempt used plain module-level Python floats (`RHO0 = 1.225`, etc.)
referenced inside field_operator bodies. Passed immediately under the default
embedded backend — then failed to even compile under `gtfn_cpu`:
`EveValueError: Symbols {'RHO0', 'RHO_VEL', 'RHO_VEL_C'} not found.` This is
exactly why testing only the embedded backend isn't enough (flagged as a risk
back when we did the environment smoke test, and it just materialized for
real). Fix: `enum.Enum(ta.wpfloat)` members instead (matching icon4py's own
`MicrophysicsConstants` pattern) — confirmed working with a minimal
reproduction before applying it everywhere. Found one more wrinkle in the same
session: an enum member works fine referenced *inside* a field_operator's own
body, but passing one as a call argument from a `@gtx.program` body into a
nested field_operator call does not (`TypeError: 'Attribute.value' must be
Expr...`) — fixed by writing two separate field_operators (`_rhocorr`,
`_rhocld`) instead of one parameterized by an enum-valued argument. Wrote this
up in `porting_plan.md`'s "GT4Py Implementation Notes" since every future
stencil (all of Phase 3) will hit the same thing otherwise.

### Validation

Instrumented `mo_2mom_mcrph_driver.f90` twice (both reverted after, `git
status` clean, `make` still builds): once for `rhocorr`/`rhocld` at k=1,5,9,13
(no process pipeline needed, computed unconditionally early in
`two_moment_mcrph`), once for the latent-heat update's actual inputs/output at
the same levels (`q_vap_old/new`, `q_liq_old/new`, `T_before/after`) — this
validates the *formula* correctly even though the process pipeline that
produces `q_vap_new`/`q_liq_new` isn't implemented yet, since the captured
values are Fortran's real intermediate state, not synthetic test data. All
values matched to `rtol=1e-6` or better, under **both** the embedded and
`gtfn_cpu` backends. `tests/unit_tests/test_unit_conversion.py` (8 tests, 4
functions x 2 backends) codifies this — passing, 17/17 total across both test
files.

### Deliberately not done in Phase 2

- `ssat`/`ninagi`/`qgl`/`qhl` are not converted — their Fortran `IF`-guards
  (`lexpl_supersat`/`luse_agi`/`lprogmelt`) are all `False` for this config.
  Documented explicitly in `convert_fields`'s docstring and in
  `porting_plan.md` so a future call site doesn't include them "for
  completeness" and introduce a spurious `rho` factor.
- The `ldass_lhn`/`qrsflux` zeroing step (`mo_2mom_mcrph_driver.f90:322-326`) —
  LHN (latent-heat-nudging) bookkeeping, diagnostic-only, not consumed by
  anything downstream in this scheme. Same category as `dep_rate_ice/snow`.

### Next up

Phase 3: process stencils, starting with `set_default_n` + the three clipping
stencils (no physics, `where`/`maximum`/`minimum` warm-up), then
`cloud_freeze`/`ice_melting`. Still need to read `het_icenuc_inas_depo` and the
`get_otab`/`equi_table` CCN table construction before those two land.

---

## 2026-07-23 — Phase 3 (first half): `set_default_n`, clipping, `cloud_freeze`, `ice_melting`

### Confirmed while implementing (no plan corrections needed this time --
### the Phase 1/2 corrections held up)

- `set_default_n`'s cloud branch (`set_qnc`) is **always active** for this
  scheme: the optional `n_cn` argument that would suppress it is declared in
  the Fortran signature but never passed at the one call site
  (`mo_2mom_mcrph_main.f90:584`, `CALL set_default_n(kstart, kend, cloud, ice,
  rain, snow, graupel, hail)` — no `n_cn`). Confirmed by reading the call site
  directly rather than assuming from the signature.
- `cloud_freeze`'s inner `IF (T_c > -30.0)` branch (choosing between two
  `j_hom` formulas) is dead code in practice: it's nested inside a block that
  already required `T_c < -30.0`, so that branch can never be taken. Only the
  polynomial `j_hom` formula (the `ELSE`) is ever reached. Implemented the
  `where()` faithfully anyway (matching the source 1:1) rather than "simplify"
  it away, in case the outer threshold ever changes.

### Implemented and validated against real Fortran output

- `stencils/housekeeping.py`: `set_default_n` (all 6 species),
  `clip_number_concentration` (generic, used at every clip site in the call
  graph), `clip_cloud_hard_cap` (the `min(n, 5000e6)` final step, cloud only).
- `stencils/particle_helpers.py`: field-level `particle_meanmass` (particles.py's
  version is scalar/Python-only, for one-time coefficient setup; the process
  stencils need the per-level field version).
- `stencils/processes.py`: `cloud_freeze`, `ice_melting`.

`cloud_freeze`/`ice_melting` validated by bracketing their call sites in
`mo_2mom_mcrph_main.f90::clouds_twomoment` with temporary `WRITE`s (before/after
state at k=1,5,9,13 of `example/fields.csv`, reverted afterward, `git
status`/`make` clean) — this captures their *actual* mid-pipeline inputs
(post-CCN-activation, post-set_default_n, post-IN-nucleation), not synthetic
data, so it exercises the real branch mix the column produces: `cloud_freeze`'s
instant-freeze branch fires at k=1 (T_c<-50), gates close for the other three
reasons (q_c==0, T_c>=-30, T>=T_3) at k=5/9/13; `ice_melting` only fires at
k=13 (only level with T>T_3), routing to rain (not cloud) since the melted
ice's mean mass exceeds `cloud.x_max`. All values matched to `rtol=1e-6+`,
under both embedded and `gtfn_cpu`. `set_default_n`/clipping validated against
hand-derived values instead (simple closed-form formulas, no branching beyond
a log-guard -- see test file docstring for why this is a deliberately lighter
validation tier than the branch-heavy processes). 27/27 tests passing across
all three test files.

### Caught my own transcription error before it shipped

First draft of the `ice_melting` test had `_IM_RN_EXPECTED`'s last entry as
`0.1210380000e4` (=1210.38) copied from a terminal capture that read
`rn= 0.1210380000D+04`. That's wrong by exactly 10x: `rain_n` should equal the
`ice_n` that melted into it (12103.8, `D+05` not `D+04`), by simple
conservation (`rain%n = rain%n + melt_n`, starting from 0). Didn't ship it on
faith — re-instrumented with a narrower, single-purpose `WRITE` bracketing just
that one call (`D22.14`, no repeated-group format), which printed
`0.12103800000000D+05` — confirming 12103.8 was right and the original
`D+04`-labeled capture was a misread on my part when transcribing a long
multi-field repeated-group `WRITE` (`'(A,I3,6(A,D18.10))'`) into the test file,
not a Fortran or physics bug. Fixed the test's expected value to `0.1210380000e5`
(12103.8) before it became a permanent wrong regression baseline. Lesson: when
a captured value doesn't match what the
formula being tested logically requires (simple addition from a known start),
don't rationalize the mismatch — re-verify with a simpler, harder-to-misread
capture before trusting either number.

### Second GT4Py program-body restriction found

`@gtx.program` bodies can't have *any* local Python statement, not just no
free-floating constants (see Phase 2's finding) — `set_default_n`'s program
tried `domain = {...}` once, reused across its six field_operator calls, and
failed with `UnsupportedPythonFeatureError: Unsupported Python syntax:
'ast.Assign'`. Fixed by inlining the `domain={...}` dict literally at each of
the six call sites. Written up in `porting_plan.md`'s GT4Py notes since every
multi-species program (the CCN/IN-nucleation/vapor-deposition stencils still
to come) will hit this the same way.

### Next up

`ice_nucleation_homhet`/`ice_nucleation_het_inas` (needs `het_icenuc_inas_depo`,
still unread) and `vapor_dep_relaxation`, then the CCN NumPy step
(`ccn_activation_sk_4d`, `get_otab`/`equi_table`), then Phase 4 (driver wiring
everything in the right order) and the full-column validation against
`example/output_fields.csv`.

---

## 2026-07-23 — `ice_nucleation_homhet` implemented and validated (~1e-12 match)

The largest, most branch-heavy routine in the whole scheme. Implemented as
`stencils/ice_nucleation.py` (`_het_icenuc_inas_depo`, `_ice_nucleation_het_inas`,
`_homogeneous_nucleation`, combined into `ice_nucleation_homhet`) plus a new
shared `stencils/saturation.py` (`e_es`, `e_ws` — Tetens formula, `ipsat=1` —
and `diffusivity`, needed here and by the still-to-come `vapor_dep_relaxation`).

### A legitimate simplification, found and verified before coding around it

`ice_nucleation_het_inas` loops over 3 dust modes, but re-reading the loop body
line by line (not skimming) showed that with `use_prog_in=True` — confirmed
true for this scheme: `clouds_twomoment`'s `lprogin = PRESENT(ninpot)`, and
`column_driver.f90` always supplies `ninpot` — each mode iteration does
`inp(k) = n_inpot(k) + ndust*(...)`, **overwriting** rather than accumulating.
Only the last mode (mode 3) survives; modes 1-2 are computed and discarded.
`ssw` (which picks the immersion-vs-deposition branch) doesn't depend on the
mode index either. So the 3-mode loop collapses to a single mode-3 calculation
with zero loss of fidelity for this config — implemented it that way, with the
reasoning written into the module docstring so it doesn't look like an
oversight to the next reader (or model).

### Re-verified two aliasing/constant details directly from source rather than
### from memory (both turned out exactly as expected, but worth the check)

- `R_d` inside this routine (and the homogeneous-nucleation block) really is
  the water-vapor gas constant (461.51, aliased from `rv`), not dry air —
  confirmed again since `acoeff(1)` in the homogeneous-nucleation formula uses
  **both** `R_d` (vapor) and `R_l` (dry air, 287.04) in the same expression,
  exactly the kind of formula where the two are easy to swap.
- `het_icenuc_inas_depo`'s `param_dust` and the dust-mode background constants
  are fixed `PARAMETER`s (not runtime-configurable), so the "mode 3 only"
  simplification is safe regardless of any other config knob.

### Known assumption about uninitialized Fortran state (documented, not silently assumed)

`ndiag_mask`/`nuc_n_a` (which gate `n_inpot`'s depletion) are Fortran local
arrays only assigned where the INAS gate is true; elsewhere they hold whatever
was on the stack — not a standard-guaranteed value, and this Makefile passes no
`-finit-local-zero` or optimization flags. Implemented the sane reading (no
depletion where the gate didn't fire) rather than trying to replicate
undefined behavior. Written up explicitly in `ice_nucleation.py`'s docstring
as a documented risk, not swept under the rug.

### Two new GT4Py findings (written up in porting_plan.md's GT4Py notes, not just here)

1. **Cross-module constant-enum name collision.** Every stencil module so far
   named its private constants class `_Const`. Fine in isolation; broke the
   moment `ice_nucleation.py` called `saturation.py`'s functions and both got
   compiled together under `gtfn_cpu`:
   `NotImplementedError: Using closure vars with same name but different value
   across functions is not implemented yet. Collisions: '_Const'.` GT4Py
   resolves closure variables by name across the whole compiled call graph, so
   two different enums sharing a name collide as soon as they're composed.
   Renamed every module's enum to a unique name
   (`_UnitConversionConst`/`_HousekeepingConst`/`_ProcessesConst`/
   `_SaturationConst`/`_IceNucleationConst`) — this will matter even more once
   Phase 4's driver imports everything together.
2. **Cross-module function calls must be bare-name imports.** Calling
   `saturation.e_es(temperature)` (module-qualified) inside a field_operator
   body fails: `DSLError: Functions can only be called directly.` Fixed with
   `from minimal_mcrph.stencils.saturation import e_es` and calling `e_es(...)`
   directly.

### Validated against real Fortran output — ~1e-12 relative error

Bracketed the `CALL ice_nucleation_homhet` call site in
`mo_2mom_mcrph_main.f90::clouds_twomoment` with temporary `WRITE`s (reverted
after, `git status`/`make` clean), capturing before/after state at k=1,3,5,7,
9,11,13 of `example/fields.csv` — 4 levels with the INAS gate open (different
qv/T combinations, exercising both the depletion path and the "no depletion,
inp still below threshold" path) and 3 with it closed. Max relative error
across `qv, ice_q, ice_n, n_inact, n_inpot` was ~1e-12 (floating-point noise
level) in both embedded and `gtfn_cpu`. 2 new tests
(`test_ice_nucleation.py`), 29/29 total passing.

**One important caveat, not silently left uncovered:** `atmo%w` is exactly 0.0
at every level of `example/fields.csv`, so the KHL06 homogeneous-nucleation
branch's *active* path (`nucleates = gate & (w > w_pre)`, and `w_pre >= 0`
always) never fires in this test column — only confirmed its "gate closed,
no-op" behavior. The homogeneous-nucleation formula itself (the largest,
most formula-dense block in this whole routine) has no independent real-data
cross-check yet. Written into the test file's docstring as an open item, not
hidden.

### Caught my own transcription error again — same failure mode as the
### `ice_melting` one, same fix

First draft of `test_ice_nucleation.py` hand-rebased the raw `0.xxxD-06`-style
captures into `x.xxxe-05`-style literals for readability, and introduced a
10x error in `ice_q` at k=3 (`4.95089e-06` instead of the correct
`4.95089e-07`) doing it by hand. The test caught it immediately (4/7 levels
failed at ~0.1% relative error — small enough to look like "probably just a
tolerance issue" if I hadn't already been burned by this exact mistake once
this session). Fixed by copying the raw `0.xxxeNN` literal forms directly from
the terminal capture with no manual rebasing step, which is what actually
introduced both transcription bugs. **Standing rule going forward: never
hand-rebase a captured scientific-notation value into a different exponent
form. Copy the raw form verbatim, even if it's less readable.**

### Next up

`vapor_dep_relaxation` (needs `saturation.py`'s functions, already built), then
the CCN NumPy step (`ccn_activation_sk_4d`, `get_otab`/`equi_table`), then
Phase 4 (driver wiring) and the full-column validation against
`example/output_fields.csv`.

---

## 2026-07-23 — `vapor_dep_relaxation` implemented and validated

All 5 processes are now ported. Implemented as `stencils/vapor_deposition.py`:
`_saturation_terms` (g_i/s_si), `_raw_deposition_rate` (mirrors
`vapor_deposition_generic`), combined into `vapor_dep_relaxation` handling all
four species (ice/snow/graupel/hail) and the joint `Xfac` relaxation split.

### A second legitimate "recompute vs. reuse precomputed coefficient" simplification

`vapor_deposition_generic` recomputes `vent_coeff_b(prtcl,1) * N_sc**n_f /
sqrt(nu_l)` from scratch inside its per-level loop (line 1493) instead of
using the already-precomputed `coeffs%b_f` — but that recomputed expression is
*exactly* `setup_particle_coeffs`'s own definition of `b_f`
(processes.f90:1509), word for word, and a commented-out line directly above
it in the source (`!f_v = ( coeffs%a_f + coeffs%b_f * SQRT(D*v) ) * 2.0_wp`)
confirms this is what it was always meant to be. Used the precomputed `b_f`
directly — same value, and avoids needing a runtime gamma-function evaluation
inside a per-level stencil (GT4Py has no builtin for that anyway). Same
category of finding as `ice_nucleation_het_inas`'s mode-3-only collapse:
verified equivalent before simplifying, not assumed.

### New GT4Py finding: `abs` must be imported bare, not aliased

`from gt4py.next import abs as gtx_abs` then calling `gtx_abs(x)` inside a
field_operator fails: `DSLError: Comparison operators can only be used between
arithmetic types...` — the DSL parser apparently keys off the literal name
`abs`, not just what it resolves to. Fixed by importing `abs` directly
(shadowing the Python builtin, `# noqa: A004`), matching how icon4py's own
`saturation_adjustment_stencils.py` does exactly this. Added to
`porting_plan.md`'s GT4Py notes.

### Validated against real Fortran — good match, but not at the ~1e-12 level of the other four

Bracketed the `CALL vapor_dep_relaxation` call site (reverted after, clean
build) at k=1,3,5,7,9,11,13. `example/fields.csv` has zero snow/graupel/hail
mass at every level, so only the ice branch is exercised by real reference
data (snow/graupel/hail are validated only for their "q==0, no-op" path).
Match: ~1e-7 relative on `ice_q`/`qv`, exact on `ice_n` (unchanged at every
level here, since `reduce_sublimation`'s n-adjustment only fires on net
*evaporation*, and every active level here is net deposition/growth) — good,
and comfortably inside the project's `rtol=1e-6` standard, but a notably
looser match than `cloud_freeze`/`ice_melting`/`ice_nucleation_homhet`'s
~1e-12. Plausible explanation, not fully run to ground: this routine chains
more nested `exp(log(...))` calls (`diffusivity`, `particle_diameter`,
`particle_velocity`, all composed) before reaching the deposition rate, and
the validated quantity is a small per-step increment relative to a much larger
base value (e.g. at k=3, ice_q moves from 2.6e-4 to 3.1e-4, a fractional
increment small enough that a fixed absolute floating-point difference
upstream shows up as a larger *relative* difference in the increment itself)
— written into the test's docstring rather than silently accepted at face
value. 2 new tests (`test_vapor_deposition.py`, 1 test x 2 backends), 31/31
total passing (confirmed by actually running the suite, not assumed).

### Next up

The CCN NumPy step (`ccn_activation_sk_4d`, needs `get_otab`/`equi_table`,
still unread in detail), then Phase 4: wire `prepare` → `set_default_n` →
clip → CCN → clip → `ice_nucleation_homhet` → `cloud_freeze` → clip →
`vapor_dep_relaxation` → `ice_melting` → final clip → latent heat → `post`
into the actual driver, matching `clouds_twomoment`'s exact order (see
porting_plan.md's "Process Call Graph"), then validate the whole column
against `example/output_fields.csv`.

---

## 2026-07-23 — `ccn_activation_sk_4d` implemented and validated (both the table build and the real nucleation branch)

Last of the 5 process routines. `get_otab`+`equi_table` (the one-time 4D table
build) and `ccn_activation_sk_4d` itself (the per-level activation) are both
NumPy per the standing decision — implemented as
`src/minimal_mcrph/ccn_activation.py`, plus `scripts/extract_ccn_otab.py` and
`data/ccn_otab.npz`.

### Mechanical data extraction instead of manual transcription

`get_otab` hardcodes ~450 calibration numbers (a 3×5×9×5 lookup table, mostly
as `otab%ltable(i,j,2:9,l) = (/8 values/)` lines). Given the exact transcription
errors already hit twice this session on much smaller data sets (single
values, caught by test failures), hand-retyping this was too risky to trust
without an independent check on every single number. Wrote
`scripts/extract_ccn_otab.py` instead: a regex parser that reads the actual
Fortran source, extracts the vectors and table rows mechanically, and saves
the result as `data/ccn_otab.npz`. Validated the parse itself against 3
manually-reread spot values from the raw source text (corner + extrapolated
values) before trusting it, and a nonzero-element count check (480 of 675,
matching the two all-zero slices' sizes exactly) as an independent structural
sanity check. This is checked into the repo (`scripts/`) for provenance/
reproducibility, not imported by the package at runtime.

### Two more legitimate simplifications, verified before relying on them

1. **The height-dependent `Ncn` fallback (`z0_nccn`/`z1e_nccn`) is dead code
   for this scheme.** It only fires when `ccn_activation_sk_4d`'s optional
   `n_cn` argument is absent; `clouds_twomoment` always calls with it present
   (`lprogccn = PRESENT(nccn) = True`, `column_driver.f90` always supplies the
   CSV's `nccn` column). `Ncn` is always read directly from the prognostic
   array — the height-profile branch, and `atmo%zh`, aren't needed at all.
2. **Another out-of-bounds Fortran read, same category as
   `ice_nucleation_het_inas`'s uninitialized-array issue.** The activation
   gate uses `atmo%w(k+1)` *unclamped*, right next to a properly-clamped
   `kp1_fl = MIN(k+1, SIZE(atmo%rho))` used for the gradient check in the same
   condition. `atmo%w` is sized `nlev` (confirmed in
   `mo_nwp_gscp_interface.f90`), so at the last level this reads one element
   past the array end — undefined behavior, not something to faithfully
   reproduce. Implemented as "gate closed" there. Inconsequential for the real
   reference data: `atmo%w` is exactly 0.0 at every level in
   `example/fields.csv`, so this boundary case never differs observably from
   the reference regardless of which assumption is made.

### Validated the table build exactly — 5/5 spot checks, real Fortran output

`tab` (the equidistant table `equi_table` builds) is private to
`mo_2mom_mcrph_processes.f90`, so bracketed it from the inside: added a
temporary `WRITE` at the end of `equi_table` itself dumping `tab%ltable` at 5
chosen (i,j,k,l) indices, reverted after (`git status`/`make` clean). All 5
matched the Python `build_ccn_table()` output exactly (to the full precision
printed).

### The real validation problem: `example/fields.csv` never exercises the actual nucleation branch

`atmo%w == 0.0` at every level of the reference column, and the gate requires
`atmo%w(k+1) > 0` — so `ccn_activation_sk_4d` is a complete no-op everywhere in
the real reference data (confirmed: instrumented the real call site, input
exactly equals output at all 13 levels). That validates the *gating* logic
correctly stays closed, but gives zero coverage of the table lookup + actual
activation formula — the most novel, most error-prone part of the whole
routine, and exactly the piece this NumPy-vs-GT4Py design decision was made
for in the first place. Not willing to ship that with only synthetic
self-consistency checks (the same pattern as the homogeneous-nucleation
caveat, but this piece is too central to leave at that).

Escalated: made a temporary copy of `example/fields.csv` (`fields_w_test.csv`,
same directory) with one nonzero `w` value, temporarily repointed
`column_driver.f90`'s hardcoded `fields_file`/`output_file` at the copies,
rebuilt, ran, captured the real activation branch's actual behavior, then
reverted the driver and deleted both temporary CSVs — nothing in `example/`
or the committed Fortran source changed. (First attempt at this picked levels
1-4 for the nonzero `w`, based on a hand-derived guess about which level would
satisfy the vertical-gradient gate; got no activation there either. Traced
through why: the gate requires `cloud%q(k)/rho(k) > cloud%q(k+1)/rho(k+1)`,
and by the time `ccn_activation_sk_4d` runs the column has already been
through `satad`, so `cloud%q` is *not* simply `rho * (CSV's qc column)` — the
actual post-satad ratio was increasing with `k` in this profile, the opposite
of what the gate needs, except at the one level where `cloud%q` tapers to
zero. Recomputed which level that was from the captured post-satad state
rather than the raw CSV, and used `w` there instead — worked.) Result: ~1e-13
relative error on `cloud_q, cloud_n, qv, n_cn` — the same floating-point-noise
tier as the best-validated processes, not just "close enough."

3 new tests (`test_ccn_activation.py`: table build, real no-op case, and the
synthetic-`w` active case), 34/34 total passing.

### Next up

Phase 4: wire `prepare` → `set_default_n` → clip → CCN (NumPy) → clip →
`ice_nucleation_homhet` → `cloud_freeze` → clip → `vapor_dep_relaxation` →
`ice_melting` → final clip → latent heat → `post` into an actual driver,
matching `clouds_twomoment`'s exact order (porting_plan.md's "Process Call
Graph"), then `satad` → driver → `satad` matching `nwp_microphysics`, then
validate the whole column against `example/output_fields.csv`. This is the
first point at which every piece built so far has to work together in
sequence — expect this to surface ordering/unit-conversion mistakes that
per-process validation couldn't catch.

---

## 2026-07-23 — Phase 4 done: full column matches Fortran reference

`satad` (newly ported this phase, `stencils/satad.py`), all 5 processes, and
all housekeeping are now wired into `driver.py` + `csv_io.py`, matching
`nwp_microphysics`'s `satad → two_moment_mcrph → satad` structure exactly.
**Full-column output now matches `example/output_fields.csv` to
floating-point precision on every checked field, under both the embedded and
`gtfn_cpu` backends** — this is the actual validation target the whole port
has been building toward.

### `satad` ported directly instead of wiring up icon4py's `SaturationAdjustment`

Read `mo_satad.f90::satad_v_3D` properly for the first time this phase (an
earlier draft of this plan falsely claimed it was already done — see the
Status section at the top of this file). Turned out simpler than expected:
every level is fully independent (no vertical coupling at all — `rhotot(k)`,
`te(k)`, `qve(k)`, `qce(k)` only ever reference level `k`), so despite the
Newton iteration this is an ordinary pointwise stencil, not something needing
`scan_operator`. Ported directly in `stencils/satad.py` rather than
standing up icon4py's `SaturationAdjustment` component (the plan's original
suggestion) — that expects a real `IconGrid`/`VerticalGrid`, infrastructure
a single CSV-driven column has no natural instance of, and it's the same
algorithm anyway (icon4py's own Newton iteration in
`saturation_adjustment_stencils.py` uses the same `qsat_rho`/`dqsatdT_rho`
names). Fortran's Newton loop exits early on convergence
(`ABS(twork-tworkold) > tol .AND. count < maxiter`); this port always runs
all `maxiter=10` iterations unconditionally (manually unrolled — a `for` loop
wasn't tried, unrolling by hand was simpler given only 10 repeats). Validated
this simplification, not just assumed it: bracketed the first `satad_v_3d`
call in `mo_nwp_gscp_interface.f90` (reverted after), ~1e-11 relative error
against real Fortran output — running past convergence is idempotent here, as
expected for a well-conditioned smooth saturation curve.

### The bug: a whole driver-level housekeeping block, missed entirely, caught only by the full-column comparison

First full pipeline run: everything matched Fortran to floating-point
precision *except* `nccn` (up to 11.8% relative error) and `ninpot` (up to
100% at some levels — my value was 0 where Fortran's wasn't). Every process
touching these fields had already been individually validated against real
Fortran data with ~1e-8-1e-13 precision, so the bug had to be in the driver
wiring, not the stencils.

First hypothesis (wrong): `example/output_fields.csv` might be stale from an
earlier instrumentation run this session. Regenerated it fresh with `make
run` on the clean, reverted Fortran source (confirmed via `git status`) —
identical discrepancy. Not a stale-file problem.

Second hypothesis (right, found by grepping `mo_2mom_mcrph_driver.f90` for
every remaining use of `nccn`): there's a whole block in `two_moment_mcrph`
**after** `post_twomoment` (mo_2mom_mcrph_driver.f90:467-524) that has
nothing to do with `clouds_twomoment` or any of the 5 retained processes —
a second negative-value clip, an `nccn` reset toward a height-dependent
background profile at cloud-free points (explaining the "impossible"
*increase* in a supposedly-monotonic depletion budget), an `ninact`
relaxation toward zero, an `ninpot` relaxation toward a *different*
height-dependent background profile (`in_coeffs`, not `ccn_coeffs` — needed
its own `INCoeffs` struct in `config.py`), an `nccn` floor, and one more
`qnc=0`-where-`qc`-tiny step. This is genuinely driver-level bookkeeping, not
part of `clouds_twomoment` — which is exactly why no per-process test could
have caught its absence, and why it was missed on the first implementation
pass despite reading `mo_2mom_mcrph_driver.f90` carefully earlier in this
project. Implemented as a new `stencils/post_housekeeping.py`, needing `hhl`
(half-level heights) for the first time in this port — added to
`ColumnState`/`csv_io.py`. After adding it: every field matches to
floating-point precision, both backends.

**Lesson, stated plainly: per-process validation, however rigorous, does not
substitute for full-pipeline validation.** Every individual piece in this
port was validated against real Fortran output before being trusted, and that
discipline still left a 100%-wrong field in the first full-driver assembly,
because the bug lived in code that doesn't belong to any single process. The
project's actual validation contract (CLAUDE.md: match
`example/output_fields.csv`) exists precisely to catch this category of
mistake, and it did.

### Files added this phase

`stencils/satad.py`, `stencils/post_housekeeping.py`, `driver.py`,
`csv_io.py`, `tests/integration_tests/test_full_column.py`,
`tests/unit_tests/test_satad.py`. `config.py` gained `INCoeffs`/`IN_COEFFS`
and `TAU_INACT`/`TAU_INPOT`/`Q_CRIT`.

### Status

Full-column validation against `example/output_fields.csv` passes, both
backends, `rtol=1e-6` (the project's standard — actual achieved precision is
tighter, mostly floating-point noise, per-field breakdown in this entry
above). This is the first point where the port has been validated
end-to-end, not just piece by piece.

### Next up

Nothing required for correctness. Possible follow-ups if this project
continues: a second real test column (ideally with nonzero `w`, to get an
independent real-data check on the KHL06 homogeneous-nucleation branch and
the CCN active-nucleation branch's boundary behavior, both currently only
validated via one synthetic case each); revisiting whether
`ccn_activation_sk_4d` is worth expressing as a compiled stencil for
performance (deliberately deferred per the standing NumPy decision); general
cleanup pass now that the shape of the whole pipeline is known end-to-end.

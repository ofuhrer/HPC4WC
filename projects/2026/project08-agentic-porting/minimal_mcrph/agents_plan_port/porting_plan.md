# Minimal 2-Moment Microphysics Port Plan (GT4Py)

## Status

**Nothing has been implemented in Python yet.** There is no `minimal_mcrph_gt4py/` (or
similarly named) package anywhere in this repository — this plan is for greenfield work.
(An earlier draft of this document claimed saturation adjustment was already ported;
that was incorrect. `mo_satad.f90` is Fortran-only today. The `SaturationAdjustment`
class this plan reuses already exists in `icon4py` — see "Icon4py Integration" — but
nothing in `minimal_mcrph` has been wired up to it.)

This plan targets the already-completed Fortran reduction documented in
[`../fortran/agents_plan/PLAN.md`](../fortran/agents_plan/PLAN.md): a single-column
scheme that runs

1. saturation adjustment (`satad_v_3d`)
2. a reduced `clouds_twomoment`:
   a. CCN activation
   b. IN nucleation (`ice_nucleation_homhet`)
   c. cloud-droplet freezing (`cloud_freeze`)
   d. ice depositional growth (`vapor_dep_relaxation`)
   e. ice-crystal melting (`ice_melting`)
3. saturation adjustment again

with no sedimentation, no collision/riming/wet-growth, no gamma-lookup tables, and no
netCDF. That Fortran reduction is **bit-identical** to the full scheme for the retained
processes — the goal of the GT4Py port is to reproduce it, not to re-approximate it.
Every process description below is taken directly from
`minimal_mcrph/fortran/mo_2mom_mcrph_processes.f90` and `mo_2mom_mcrph_main.f90`, with
line numbers, so it can be checked against the source as it changes.

## Default Configuration (`column_driver.f90`, `igscp=5`)

**Correction (2026-07-23): an earlier version of this plan claimed `nuc_i_typ=6`
(Phillips 2010). That was wrong** — re-derived from the actual `cloud_type` arithmetic
below and confirmed against the code path it selects, the default column-driver run
exercises `nuc_i_typ=1` (`ice_nucleation_het_inas`, Ullrich et al. 2007), not Phillips.
Do not implement Phillips as the default target; see the corrected table and reasoning
below. (Caught before any code was written against the wrong value — see `port_log.md`.)

`column_driver.f90` calls `two_moment_mcrph_init(igscp=5, ice_type=1, N_cn0=..., ...)`
with no `cfg_2mom` override, so `cfg_params = cfg_2mom_default`
(`mo_2mom_mcrph_config_default.f90`). `two_moment_mcrph_init` then derives (lines
684-705 of `mo_2mom_mcrph_driver.f90`):

```
ccn_type   = ccn_type_gscp5 = 8                      (cfg_2mom_default%ccn_type = -1, not > 0, so the gscp5 default wins)
cloud_type = cloud_type_default_gscp5 + 10*ccn_type  = 2003 + 80  = 2083
cloud_type = cloud_type + 100*ice_type               = 2083 + 100 = 2183
```

and `init_2mom_scheme_once` (`mo_2mom_mcrph_main.f90:751-754`) decodes that single
integer into the four scheme-selection flags:

```
ice_typ   = cloud_type / 1000        = 2      ! "with hail" — unused by anything in the retained call tree, safe to ignore
nuc_i_typ = MOD(cloud_type/100, 10)  = 1      ! -> ice_nucleation_het_inas, NOT Phillips
nuc_c_typ = MOD(cloud_type/10, 10)   = 8      ! -> ccn_activation_sk_4d (this part was already correct)
auto_typ  = MOD(cloud_type, 10)      = 3      ! warm-rain scheme choice — unused (no autoconversion in this scheme)
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `igscp` | 5 | 2-moment scheme w/ prognostic CCN+IN |
| `ice_type` (input) | 1 | passed to `two_moment_mcrph_init`; feeds `cloud_type` above — **do not confuse with `nuc_i_typ`**, they're different numbers |
| `luse_agi` | `.FALSE.` | AgI (cloud-seeding) tracer path disabled |
| `lexpl_supersat` | `.FALSE.` | no explicit supersaturation prognostic variable |
| `nuc_c_typ` | 8 | CCN activation: **Segal & Khain (2006) 4D lookup table**, `ccn_activation_sk_4d`, **continental** aerosol case (see below) |
| `nuc_i_typ` | 1 | IN nucleation: **Ullrich et al. (2007) INAS** heterogeneous scheme, `ice_nucleation_het_inas`, plus homogeneous (KHL06) nucleation (nuc_i_typ in `1:9` always turns homogeneous on) |

**CCN aerosol case — continental, not maritime.** `ccn_type=8` alone doesn't fully
determine the CCN coefficients: `two_moment_mcrph_init` (`mo_2mom_mcrph_driver.f90:
747-762`) additionally branches on `cfg_params%tune_sbmccn` (default `1.0`,
`mo_2mom_mcrph_config_default.f90:42`). Since `1.0` is not `< 1.0`, the **continental**
branch is taken: `Ncn0=1700e6, Nmin=35e6, lsigs=0.2, R2=0.03, etas=0.7`
(not the maritime numbers `100e6/35e6/0.4/0.03/0.9` from the same `CASE(8)` block's
other branch — easy to grab the wrong one by only reading the `CASE(8)` label). Plus,
unconditionally: `z0=4000, z1e=2000` (`mo_2mom_mcrph_driver.f90:723-724`),
`wcb_min=cfg_params%ccn_wcb_min=0.1` (line 727, `mo_2mom_mcrph_config_default.f90:41`).

**Particle-type assignment.** `init_2mom_scheme` (`mo_2mom_mcrph_main.f90:663-721`)
assigns fixed named parameter sets, and every per-species override in
`cfg_2mom_default` is the `-999.99` sentinel (= "don't override"), so for the default
column-driver run these are used completely unmodified: `cloud=cloud_nue1mue1` (lines
298-315), `rain=rainSBB` (436-453), `ice=ice_cosmo5` (317-340), `snow=snowSBB`
(367-390), `graupel=graupelhail_cosmo5` (170-193, since `igscp≠7` selects the
`particle_frozen` branch, not the LWF/`graupel_vivek` one), `hail=hail_cosmo5`
(225-248, same LWF/non-LWF reasoning).

**For the default column-driver configuration, only these process bodies actually run:**
`ccn_activation_sk_4d`, `ice_nucleation_het_inas` + the homogeneous-nucleation block
inside `ice_nucleation_homhet`, `cloud_freeze`, `vapor_dep_relaxation`, `ice_melting`.
`ccn_activation_hdcp2`, `ice_nucleation_het_philips`, `ice_nucleation_agi_dm95/m16` are
kept in the Fortran (per `agents_plan/PLAN.md`'s keep-list) but are **not exercised**
unless `nuc_c_typ<6`, `nuc_i_typ≥5`, or `luse_agi=.TRUE.` — they don't need to be ported
for the port to reproduce `example/fields.csv` → `example/output_fields.csv`, only
implemented later if those config knobs are exposed.

### CCN activation: implement the Segal & Khain 4D LUT, not a Hande substitute

An earlier draft of this plan proposed implementing the simpler Hande et al. formula
(`ccn_activation_hdcp2`) first and deferring the actual configured Segal & Khain LUT
scheme "for later." **Don't do this.** `nuc_c_typ=8` is what the reference CSV output
was generated with, and the LUT (`get_otab`/`equi_table`,
`mo_2mom_mcrph_processes.f90:1641-1809`) needs neither incomplete-gamma tables nor
netCDF — it's exactly the kind of thing the stripped Fortran scheme kept because it's
tractable. Substituting Hande would silently change the physics and make the GT4Py
output diverge from the Fortran reference by construction, not by a numerical bug.
`ccn_activation_hdcp2` is fine as an early smoke-test stencil (it's simpler code to get
GT4Py idioms right on) but must not be mistaken for, or substituted for, the actual
target.

---

## Physical units: mixing ratios vs. densities — do not skip this

The CSV columns (`qc, qnc, qr, ...`) and the process subroutines are **not** in the same
units. `two_moment_mcrph` (`mo_2mom_mcrph_driver.f90:379-465`) sandwiches the call to
`clouds_twomoment` between two unit conversions:

```
prepare_twomoment:  q  *= rho     (mixing ratio  [kg/kg]      -> mass density  [kg/m3])
                     n  *= rho     (conc.         [1/kg]        -> number density [1/m3])
...
clouds_twomoment operates entirely on densities...
...
post_twomoment:     q  *= 1/rho   (mass density   -> mixing ratio)
                     n  *= 1/rho   (number density -> conc.)
```

(`mo_2mom_prepare.f90:47-90` and `:229-274`; `1/rho` is precomputed once as `rho_r` in
`mo_2mom_mcrph_driver.f90:358`.) All five retained process routines
(`ccn_activation_*`, `ice_nucleation_homhet`, `cloud_freeze`, `vapor_dep_relaxation`,
`ice_melting`) and `set_default_n` operate on **densities**. A GT4Py port that runs these
stencils directly on the CSV's mixing-ratio columns without this `*rho` / `/rho`
bracketing will be wrong by a factor of `rho` (~1 kg/m³) throughout — small enough to
look "roughly right" in a quick check, wrong enough to fail any `rtol=1e-6` comparison.
Port `prepare_twomoment`/`post_twomoment` (or their density-conversion effect) as an
explicit stencil pair, not an implementation detail to remember later.

### Latent-heat temperature update

**Correction (2026-07-23): the default config uses temperature-dependent latent
heat, not the constant `als`/`alv-als` branch** — re-derived instead of leaving
this as an "or" between two options. `nwp_gscp_interface.f90` passes `l_cv=.TRUE.`
unconditionally, and `column_driver.f90` sets `ithermo_water=1`. In
`two_moment_mcrph` (`mo_2mom_mcrph_driver.f90:267-271`):
`lconstant_lh = (ithermo_water == 0)` → `(1 == 0)` → **`.FALSE.`** — so the
temperature-dependent branch (`latent_heat_sublimation`/`latent_heat_melting`,
both from `mo_satad.f90`) is the one actually exercised, not the constant one.

After `clouds_twomoment` runs (still in density space) and before `post_twomoment`,
`two_moment_mcrph` updates temperature from the pre/post water-vapor and liquid-water
densities (`mo_2mom_mcrph_driver.f90:392-450`):

```python
# per level, using densities q_vap/q_liq before and after clouds_twomoment
led = latent_heat_sublimation(T)  # mo_satad.f90:634-650 -- see formula below
lwe = latent_heat_melting(T)      # mo_satad.f90:652-669
z_heat_cap_r = 1.0 / cv_d         # cv_d = cpd - rd = 717.6; l_cv=True is always passed here
convice = z_heat_cap_r * led
convliq = z_heat_cap_r * lwe
dT = -convice * rho_r * (qv_new - qv_old) + convliq * rho_r * (q_liq_new - q_liq_old)
T += dT
```

```python
# mo_satad.f90:612-669, using r_v=461.51 (water vapor gas constant),
# cp_v=1850 (specific heat of water vapor), ci=2108 (specific heat of ice),
# cl=clw=(3.1733+1)*cpd=4192.664 (specific heat of liquid water),
# lwd=alv=2.5008e6, led_const=als=2.8345e6, tmelt=273.15
latent_heat_sublimation(T) = als + (cp_v - ci) * (T - tmelt) - r_v * T
latent_heat_melting(T)     = alv - als + (ci - cl) * (T - tmelt)
```

where `q_liq = qc + qr` (no `lprogmelt` in the minimal scheme, so no `qgl`/`qhl` term).
This is the mechanism by which `cloud_freeze`/`vapor_dep_relaxation`/`ice_melting`
changing `qv`/`qc`/`qr` feeds back into `T` — it must be ported as its own step between
the process chain and the final unit conversion, not folded into any single process.

### Density-correction factors for terminal fall velocity (`rhocorr`, `rhocld`)

Computed once per level in `two_moment_mcrph` right before `prepare_twomoment`
(`mo_2mom_mcrph_driver.f90:355-365`), **not** inside `prepare_twomoment` itself:

```python
rho_r    = 1.0 / rho
rhocorr  = exp(-rho_vel   * log(max(rho, 1e-6) / rho0))   # rho_vel   = 0.4
rhocld   = exp(-rho_vel_c * log(max(rho, 1e-6) / rho0))   # rho_vel_c = 0.2, rho0 = 1.225
```

`prepare_twomoment` then attaches these as each hydrometeor's `rho_v` pointer
(`mo_2mom_prepare.f90:110-115`): `cloud.rho_v = rhocld`; `rain/ice/snow/graupel/
hail.rho_v = rhocorr` (all five share the same array). This matters because
`vapor_deposition_generic` (the routine `vapor_dep_relaxation` calls per species,
`mo_2mom_mcrph_processes.f90:1466-1500`) computes its ventilation term as
`v = particle_velocity(p, x) * p.rho_v(k)` — **`rho_v` is a real input to the one
process that needs it, not just driver bookkeeping.** An earlier pass over this
plan's process-interface table for `vapor_dep_relaxation` didn't call this out
explicitly; make sure the Phase-3 implementation multiplies by the right `rho_v`
per species (`rhocld` for cloud — unused here since `vapor_dep_relaxation` doesn't
touch cloud, but the same field exists for other processes' potential future use —
`rhocorr` for ice/snow/graupel/hail).

### Negative-mixing-ratio clip, and the field list for `prepare`/`post`

Two more details easy to drop silently:

- **Before** the density conversion (i.e. before `prepare_twomoment` is even
  called), `two_moment_mcrph` clips `qr, qi, qs, qg, qh` (not `qc`, not `qv`) to
  `max(q, 0)` in mixing-ratio space (`mo_2mom_mcrph_driver.f90:312-318`).
- The exact field list that gets `*= rho` / `*= rho_r`, for this scheme's actual
  present/absent optional arguments (`lprogccn=lprogin=True` since `column_driver.f90`
  always supplies `nccn`/`ninpot`; `luse_agi=lexpl_supersat=lprogmelt=False`
  — see `mo_2mom_mcrph_driver.f90:261-263`): `qv, qc, qnc, qr, qnr, qi, qni, qs,
  qns, qg, qng, qh, qnh, ninact, nccn, ninpot` (16 fields). `ninagi`, `ssat`,
  `qgl`, `qhl` are **not** converted for this config (their `IF` guards are all
  false) — don't multiply them by `rho`/`rho_r` even though they're present as
  CSV columns / driver state, or they'll silently pick up a spurious `rho` factor.

---

## Package Layout

```
minimal_mcrph_gt4py/
├── pyproject.toml
├── README.md
├── src/minimal_mcrph/
│   ├── __init__.py
│   ├── constants.py            # physical + scheme constants not already in icon4py.model.common.constants
│   ├── config.py               # MicrophysicsConfig (nuc_c_typ, nuc_i_typ, luse_agi, iagi_param, ...)
│   ├── particles.py            # particle-type constants (x_min/x_max/a_geo/... per hydrometeor) + derived coeffs
│   ├── satad_driver.py         # thin wrapper around icon4py's existing SaturationAdjustment
│   ├── stripped_mcrph_driver.py  # prepare -> processes -> latent heat -> post, matching two_moment_mcrph
│   └── stencils/
│       ├── __init__.py
│       ├── processes.py        # ccn_activation_sk_4d, ice_nucleation_homhet, cloud_freeze,
│       │                       # vapor_dep_relaxation, ice_melting, set_default_n, clipping
│       └── unit_conversion.py  # prepare/post density<->mixing-ratio stencils + latent-heat update
└── tests/
    ├── conftest.py
    ├── fixtures.py
    └── integration_tests/
        ├── test_saturation_adjustment.py
        └── test_stripped_mcrph.py   # round-trips example/fields.csv through the Fortran binary and compares
```

## Fortran → Python File Mapping

| Fortran | Python | Notes |
|---|---|---|
| `mo_2mom_mcrph_types.f90` | `particles.py` | particle-type constants (see Coefficients section) |
| `mo_2mom_mcrph_config*.f90` | `config.py` | scheme selection flags |
| `mo_satad.f90` | `satad_driver.py` | reuse `icon4py`'s existing `SaturationAdjustment`, don't reimplement |
| `mo_2mom_prepare.f90` | `stencils/unit_conversion.py` | density conversion — see units section above |
| `mo_2mom_mcrph_processes.f90` (keep-list subset) | `stencils/processes.py` | see interface tables below |
| `mo_2mom_mcrph_main.f90::clouds_twomoment` | `stripped_mcrph_driver.py` | process ordering + clipping |
| `mo_2mom_mcrph_driver.f90::two_moment_mcrph` | `stripped_mcrph_driver.py` | prepare → clouds_twomoment → latent heat → post |
| `mo_nwp_gscp_interface.f90::nwp_microphysics` | `stripped_mcrph_driver.py` (top level) | satad → two_moment_mcrph → satad |

---

## Process Call Graph (`clouds_twomoment`, `mo_2mom_mcrph_main.f90:557-651`)

In order, with the exact clipping calls interleaved (this ordering and interleaving is
load-bearing — see "Process Ordering" below):

1. **CCN activation** — `ccn_activation_sk_4d` (line 575) updates `cloud%q, cloud%n, atmo%qv`
2. `set_default_n` (line 584) — fills in `n` from `q` for any hydrometeor still at its
   default (see exact formulas below); for cloud, only runs when *not* using prognostic
   CCN (`n_cn` absent)
3. clip `cloud%n` to `[q/x_max, q/x_min]`, only if `nuc_c_typ != 0` (lines 587-590)
4. **IN nucleation** — `ice_nucleation_homhet` (line 596) updates `ice%q, ice%n, atmo%qv`,
   and `n_inpot` if `use_prog_in`
5. **Cloud freezing** — `cloud_freeze` (line 600) updates `cloud%q, cloud%n, ice%q, ice%n`
6. clip `ice%n` to `[q/x_min, q/x_max]` (lines 603-606, note: min-then-max here, unlike
   step 3 which is max-then-min — same result, just written in the opposite order)
7. **Vapor deposition** — `vapor_dep_relaxation` (line 613) updates
   `ice%q, ice%n, snow%q, snow%n, graupel%q, graupel%n, hail%q, hail%n, atmo%qv,
   dep_rate_ice, dep_rate_snow`
8. **Ice melting** — `ice_melting` (line 621) updates `ice%q, ice%n, cloud%q, cloud%n,
   rain%q, rain%n`
9. final clipping for **all** six hydrometeors (lines 630-649): `n = min(n, q/x_min)`,
   then `n = max(n, q/x_max)`; for cloud only, additionally `n = min(n, 5000d6)` — this
   hard cap is easy to drop by accident, don't.

`dep_rate_ice`/`dep_rate_snow` are only consumed by riming/conversion routines that this
minimal scheme drops — port them as outputs for interface fidelity, but they're dead
ends here.

### Driver-level housekeeping *outside* `clouds_twomoment` — easy to miss entirely

**This one was actually missed** in the first full-driver implementation pass
(caught by the full-column comparison against `example/output_fields.csv`
showing `nccn` and `ninpot` off by up to 100% — every individual process had
already validated correctly in isolation, because this housekeeping isn't
part of any of them). `two_moment_mcrph` (`mo_2mom_mcrph_driver.f90:467-524`)
runs a block **after** `post_twomoment` (so back in mixing-ratio space, not
density) that has nothing to do with `clouds_twomoment`:

1. A second negative-value clip on `qr,qi,qs,qg,qh` and `qnr,qni,qns,qng,qnh`
   (lines 474-487) — redundant with the pre-`prepare` clip in most cases, but
   cheap to include for fidelity.
2. **`nccn` reset at cloud-free points** (lines 489-500): where `qc <=
   q_crit(=1e-9)`, `nccn = max(nccn, background_profile)` — a height-dependent
   profile using `ccn_coeffs%z0/z1e/Ncn0`, the same coefficients as CCN
   activation. This can only ever *increase* `nccn`, which is why it looked
   like a real bug when first found (a supposedly-monotonic depletion budget
   coming back higher than it went in).
3. **`ninact` relaxation toward zero** where `qi == 0` (lines 502-505), time
   constant `tau_inact=600s`.
4. **`ninpot` relaxation toward a height-dependent background profile**
   (lines 507-516, unconditional — not gated on any `q`), using
   `in_coeffs%N0=200e6, z0=3000, z1e=1000` (a *different* coefficient set from
   `ccn_coeffs`, needs its own `INCoeffs` struct), time constant
   `tau_inpot=1800s`.
5. `nccn` floored at `35e6` (line 519).
6. `qnc = 0` where `qc < 1e-12` (line 522) — same `zero_n_where_q_tiny` shape
   as `prepare_twomoment`'s housekeeping, just once more here.

This needs `hhl` (half-level heights) to compute `zf = 0.5*(hhl(k)+hhl(k+1))`
— the only place in the whole retained call tree that needs it. Implement
this as its own step in the driver, run once per timestep after `post`,
before the final `satad` call — don't fold it into `clouds_twomoment`'s
housekeeping, it isn't part of that subroutine.

---

## Process Interfaces (as actually implemented)

All fields below are **densities** (see units section) unless noted. `atmo` fields are
`w, p, T, rho, qv, zh, tke` (all per-level 1-D pointers into the column,
`mo_2mom_mcrph_types.f90:31-33`).

### `ccn_activation_sk_4d(kstart, kend, ccn_coeffs, atmo, cloud, n_cn)` — lines 1641-1809 — **implemented and validated**, `src/minimal_mcrph/ccn_activation.py`

- Inputs: `atmo%w` (updraft at the lower cell face, `atmo%w(k+1)`), `atmo%qv`,
  `atmo%rho`, `cloud%q`, `cloud%n`, `n_cn` (prognostic CN number density; always
  present for this config — see below), plus the scalar `ccn_coeffs` fields
  `Ncn0, etas, wcb_min, R2, lsigs` and the module-level 4D table `tab` (see
  Coefficients). **`atmo%zh` is not actually needed**: it only feeds the
  height-dependent `Ncn` fallback, which is dead code for this config (next bullet).
- **Simplification, not an approximation:** the height-dependent `Ncn` fallback
  (`z0_nccn`/`z1e_nccn`) is only reached when the optional `n_cn` argument is
  *absent*. `clouds_twomoment` always calls with `n_cn` present
  (`lprogccn = PRESENT(nccn) = True`, since `column_driver.f90` always supplies
  the CSV's `nccn` column) — so `Ncn` is always read directly from the
  prognostic array; the height-profile branch isn't implemented.
- Outputs: `cloud%q, cloud%n, atmo%qv`, and `n_cn` decremented by the activated number.
- Gate: only activates where `wcb > 0`, where `wcb` is `atmo%w(k+1)` if the in-cloud
  vertical gradient condition holds (`cloud%q(k)/rho(k) > cloud%q(k+1)/rho(k+1)`, or
  `k` is the top level), else `0`.
- `atmo%rho` and `atmo%temp` are **not** the only inputs — do not port this from a
  temperature/density-only signature.
- **Known assumption about an out-of-bounds Fortran read:** `atmo%w(k+1)` is used
  unclamped (unlike the `kp1_fl`-clamped gradient check right next to it), and
  `atmo%w` is sized `nlev` — so at the last level this reads one element past
  the array end. Implemented as "no data past the end → gate closed" there,
  same category of decision as `ice_nucleation_het_inas`'s uninitialized
  `ndiag_mask`. Inconsequential for `example/fields.csv` (`atmo%w=0` everywhere
  real, so this boundary case is never observably different from the reference).
- The ~450-number hardcoded `get_otab` lookup table was extracted mechanically
  from the Fortran source via `scripts/extract_ccn_otab.py` (regex parse + spot
  checks), not hand-transcribed — see that script and `data/ccn_otab.npz` for
  provenance. `equi_table`'s tetra-linear interpolation onto the equidistant
  3×5×129×11 grid is ported as plain NumPy in `build_ccn_table()`, validated
  against 5 real `tab%ltable` values from an instrumented Fortran run (exact
  match). The per-level activation logic was validated two ways: against real
  `example/fields.csv` data (a no-op there, since `atmo%w=0` everywhere — only
  confirms the gate stays closed) and against a temporary modified copy of that
  CSV with one nonzero `w` fed through the real Fortran binary (~1e-13 relative
  error on the actual nucleation branch — see port_log.md, 2026-07-23).

**Implementation decision: this one is NumPy, not a `@gtx.field_operator`, for the
first working version.** Lines 1770-1784 gather from `tab%ltable` using indices
(`iu, ju, ku, lu`) computed at runtime from `r2, lsigs` (constants) and `ncn, wcb`
(per-level field values) — a data-dependent multi-dimensional gather. That doesn't fit
GT4Py's elementwise/neighbor-offset stencil model, and forcing it into one (e.g. as a
`scan_operator`) doesn't actually solve the indexing problem, it just relocates it.
Do the whole routine — the one-time `get_otab`/`equi_table` table construction *and*
the per-step interpolation — as plain NumPy in the driver: pull `cloud.q, cloud.n,
atmo.w, atmo.zh, atmo.qv, n_cn` out of their `Field`s as arrays, run the lookup/gather
with NumPy indexing exactly as the Fortran does, and wrap the updated `cloud.q,
cloud.n, atmo.qv, n_cn` back into `Field`s (`gtx.as_field(..., allocator=backend)`)
before the next `@gtx.program` call. This is a supported seam, not a workaround: only
code inside `@gtx.field_operator`/`@gtx.program` is compiled/backend-restricted (and
that includes under `gtfn_cpu`), so an ordinary Python function handing `Field`s to and
from the surrounding pipeline works the same way regardless of backend — it's exactly
how a driver already stitches together separate compiled programs. The cost is that
this one step doesn't benefit from gtfn compilation and would force a host/device
round-trip on a GPU backend — irrelevant for a single column on CPU now, worth
revisiting only if/when performance work on a full 3D grid makes it matter. Get
correctness first; a fused/compiled version of this lookup (e.g. unrolling the small
fixed `r2 × lsigs` grid into `where` cascades) is future optimization work, not a
blocker for the initial port.

### `ccn_activation_hdcp2(kstart, kend, atmo, cloud)` — lines 1563-1638 (not on the default path; smoke-test only)

- Inputs: `atmo%p (pressure)`, `atmo%w`, `cloud%q`, `atmo%qv`.
- Formula: 4 arctangent fits in pressure give `acoeff/bcoeff/ccoeff/dcoeff`, then
  `nuc_n = acoeff * atan(bcoeff*log(wcb) + ccoeff) + dcoeff`, gated on `q_c > eps` and
  `wcb = atmo%w(k) > 0`, floored at `1e7`. There are 16 hardcoded constants
  (`a_ccn/b_ccn/c_ccn/d_ccn`, lines 1584-1592) — this is **not** an exponential-in-`qv*rho`
  formula; don't invent a different functional form for this.

### `ice_nucleation_homhet(kstart, kend, use_prog_in, atmo, cloud, ice_in, n_inact, n_inpot, n_inagi, luse_agi, iagi_param)` — lines 561-823

- `n_inact` is **mandatory** in the Fortran signature (no `OPTIONAL`), and always
  supplied from the CSV's `ninact` column by `column_driver.f90` — don't model it as
  optional/secondary.
- `luse_agi`/`iagi_param` are declared `OPTIONAL` but dereferenced unconditionally
  inside (`IF (luse_agi .AND. use_prog_in)` at line 702) — treat them as always-required
  in practice; for the default config `luse_agi=.FALSE.` so the AGI branch
  (`ice_nucleation_agi_dm95/m16`) and `ice_nucleation_het_philips` are dead at runtime,
  but the port still needs real (constant `False`/dummy) values wired through, not
  `None`.
- **Default-config path is `nuc_i_typ=1` → `ice_nucleation_het_inas`** (see "Default
  Configuration" above for how `nuc_i_typ=1` is derived — an earlier draft of this plan
  had this as `nuc_i_typ=6`/Phillips, which was wrong), dispatched at line 712-714
  (`nuc_typ < 5` branch), then always runs the homogeneous KHL06 nucleation block
  (lines 748-820) since `nuc_typ` in `1:9` sets `use_homnuc=.TRUE.`.

### `ice_nucleation_het_inas(kstart, kend, atmo, cloud, ice, ninact, inuc, use_prog_in, n_inpot, ndiag_mask, nuc_n_a)` — lines 826-996 (the default heterogeneous-nucleation path)

Ullrich et al. (2007) INAS (ice nucleation active surface site) parameterization, called
with `inuc = nuc_i_typ = 1` for the default config, giving `sfactor = 10**(inuc-1) = 1`
(scales three fixed background dust modes — this is a no-op multiplier at `inuc=1`, but
don't hardcode it away, `nuc_i_typ` is a real config knob).

- Per level: `ssi = qv*T*R_d/e_es(T) - 1` (supersaturation over ice), gated on
  `(ssi > 0.02 or cloud%q > 1e-20) and 190 < T < 265`.
- For each of 3 fixed dust modes (background number/diameter/sigma constants, lines
  865-868): computes `ssw` (supersaturation over water via `e_ws(T)`); if
  `ssw > 0.99 and T > 235`, immersion-freezing INAS density
  `inas = exp(151.548 - 0.521*T)`; else deposition-nucleation INAS density via
  `het_icenuc_inas_depo(T, ssi, param_dust)` (lines ~1000+, not yet transcribed here —
  read it directly before implementing, don't guess the functional form the way an
  earlier draft did for a different routine). Both accumulate into a per-level `inp`
  (or, if `use_prog_in`, replace the prognostic-INP-based term instead of accumulating).
- `nhet = min(inp, ni_het_max)`, `nuc_n = max(nhet - n_inact, 0)` (note: uses the
  **prognostic** `n_inact`, not `ice%n`, per `use_ninact=.true.` hardcoded at line 879),
  `nuc_q = min(nuc_n*ice%x_min, atmo%qv)`, `nuc_n = nuc_q/ice%x_min`.
- Outputs: `ice%q, ice%n` (incremented), `atmo%qv` (decremented), **`n_inact` itself is
  incremented** (`ninact(k) = ninact(k) + nuc_n`, line 973) — this is a running budget
  across the whole column driver, not a per-call scratch variable, and it's a different
  output than the shared `ice_nucleation_homhet` outputs table below implies; also sets
  `nuc_n_a`/`ndiag_mask` which `ice_nucleation_homhet` uses afterward to deplete
  `n_inpot`.

`ice_nucleation_homhet` (the outer routine) outputs overall: `ice%q, ice%n, atmo%qv,
n_inact` (incremented by `ice_nucleation_het_inas`), and `n_inpot` decremented (only if
`use_prog_in`).

### `cloud_freeze(kstart, kend, dt, cloud_coeffs, qnc_const, atmo, cloud_in, ice)` — lines 486-558

- Inputs: `dt`, `cloud_coeffs%c_z` (only `c_z` is used, not `a_f`/`b_f`), `qnc_const`
  (only matters when `nuc_c_typ==0`, i.e. not the default config), `atmo%T`, `cloud%q`,
  `cloud%n`, `ice%n`.
- Gate: `T < T_3` and `T-T_3 < -30`. Below `-50°C`: instantaneous complete freezing.
  Above: Jeffrey & Austin (1997) homogeneous freezing rate `j_hom` (two different
  polynomial branches at `T_c=-30`), `fr_n = j_hom*q_c*dt`, `fr_q = j_hom*q_c*x_c*dt*c_z`
  where `x_c = particle_meanmass(cloud, q_c, n_c)`.
- Outputs: `cloud%q, cloud%n` (decremented), `ice%q, ice%n` (incremented by the same
  `fr_q, fr_n`, with `fr_n` re-floored at `fr_q/cloud%x_max` after the cloud decrement).

### `vapor_dep_relaxation(kstart, kend, dt, ice_coeffs, snow_coeffs, graupel_coeffs, hail_coeffs, atmo, ice_in, snow_in, graupel_in, hail_in, dep_rate_ice, dep_rate_snow)` — lines 1319-1463

- Inputs: `atmo%p, atmo%T, atmo%qv`, and per hydrometeor `q, n` plus its
  `particle_coeffs` (`a_f, b_f, c_i` via `vapor_deposition_generic`, lines 1466-1500).
- **`qvsi` (saturation vapor density over ice) is computed internally, not passed in** —
  `e_si = e_es(T_a)` then `qvsidiff = atmo%qv(k) - e_si/(R_d*T_a)` (line 1393; `e_es` is
  the saturation-vapor-pressure-over-ice function from `mo_satad.f90`'s dependencies).
  An earlier draft of this plan listed `qvsi` as an input field — don't add that field;
  compute it the same way `vapor_dep_relaxation` does.
  `s_si` (supersaturation, used to weight each species' raw deposition rate before the
  relaxation split) is also computed internally per-level from `e_d = atmo%qv*R_d*T_a`.
- Relaxation-time-scale deposition (Morrison/Curry/Khvorostyanov): computes an
  unconstrained deposition rate per species via `vapor_deposition_generic`, then
  rescales all four rates jointly by `Xfac = qvsidiff/Xi_i * (1 - exp(-dt*Xi_i))` where
  `Xi_i` is the sum of the four species' relaxation rates, so the total doesn't
  overshoot `qvsidiff` within one step.
- On net evaporation (`dep < 0`), also reduces `n` (not just `q`) via
  `dep_n_fac=0.5` times `dep/particle_meanmass(...)`, floored at 0 — this "reduce
  sublimation" branch is easy to miss since it only fires on the negative-rate path.
- Outputs: `ice%q, ice%n, snow%q, snow%n, graupel%q, graupel%n, hail%q, hail%n, atmo%qv,
  dep_rate_ice, dep_rate_snow`.

### `ice_melting(kstart, kend, atmo, ice_in, cloud, rain)` — lines 1515-1560

- Gate: `T > T_3 and ice%q > 0`. Complete melt within one step (no partial melting).
- Routing: melted mass/number goes to **either** `rain` or `cloud` depending on
  `particle_meanmass(ice, q_i, n_i) > cloud%x_max` — i.e. large ice crystals melt into
  rain-sized drops, small ones into cloud droplets. Both `cloud` and `rain` are
  plausible outputs; a port that always routes to one or the other is wrong.

### `set_default_n(kstart, kend, cloud, ice, rain, snow, graupel, hail, n_cn)` — lines 2065-2116

Fills in `n` wherever `q > 0 and n < 1e-3`, using a **different formula per
hydrometeor** (from `mo_2mom_mcrph_util.f90`), not a single `q/x_max_default`:

| Hydrometeor | Formula | Notes |
|---|---|---|
| cloud | `set_qnc(qc) = qc * 6/(pi*rho_w*Dmean^3)`, `Dmean=10µm` | **skipped entirely** if `n_cn` is present (i.e. under prognostic CCN / the default config) |
| ice | `set_qni(qi) = qi / 1e-10` | |
| rain | `set_qnr(qr) = N0r*(qr*6/(pi*rho_w*N0r*Γ(4)))^0.25`, `N0r=8e6` | |
| snow | `set_qns(qs)` similarly from an exponential-PSD `N0s, ams, bms` | |
| graupel | `set_qng(qg)` similarly, `N0g, amg, bmg` | |
| hail | `set_qnh_expPSD_N0const(qh, 750, 1e6)` | fixed bulk density 750 kg/m³, `N0=1e6` |

---

## Coefficients

Two categories, don't conflate them:

**1. Static per-hydrometeor constants** (`particle`/`particle_frozen` fields in
`mo_2mom_mcrph_types.f90:36-61`; numeric values in the big parameter tables of
`mo_2mom_mcrph_main.f90`, e.g. lines ~140-260 for cloud/rain/ice/snow): `x_min, x_max,
a_geo, b_geo, a_vel, b_vel, a_ven, b_ven, nu, mu, cap`. These are compile-time constants
per hydrometeor type, ported as-is (a small `dataclass`/module constants per particle
type — no computation needed).

**2. Derived coefficients**, computed once from the above via two routines in
`mo_2mom_mcrph_processes.f90`:

```python
# setup_particle_coeffs (line 1503) — pure functions of the static constants,
# no vertical-field dependence, computed once at start-up per hydrometeor:
c_i = 1.0 / cap
a_f = vent_coeff_a(ptype, 1)                          # line 410, uses nu, mu, b_geo, a_ven
b_f = vent_coeff_b(ptype, 1) * N_sc**n_f / sqrt(nu_l)  # line 423, uses nu, mu, b_geo, b_vel, b_ven
                                                        # N_sc=0.710, n_f=0.333 (line 157-158), nu_l=kinematic viscosity
c_z = moment_gamma(ptype, 2)                           # line 440, uses nu, mu

# init_2mom_sedi_vel (line 451) — none of its three outputs are read anywhere
# in the retained call tree (grep confirms coeff_alfa_n/q/coeff_lambda are only
# written here and printed in an isprint debug block, mo_2mom_mcrph_processes.f90:
# 457-465) — they only ever fed sedimentation, which this scheme doesn't have.
# particle_diameter/particle_velocity (used by the retained processes) read the
# particle's own a_geo/b_geo/a_vel/b_vel directly, NOT these coefficients. An
# earlier draft of this plan claimed coeff_lambda was still used — it isn't.
# Worth porting anyway, for one reason only: it's the one coefficient set the
# Fortran already prints via `isprint` with zero modification needed (see
# "Testing Strategy"), so it's a free, real validation point for the gamma-function
# math shared with setup_particle_coeffs above, even though the port's actual
# physics never consumes it.
coeff_alfa_n = a_vel * Γ((nu+b_vel+1)/mu) / Γ((nu+1)/mu)
coeff_alfa_q = a_vel * Γ((nu+b_vel+2)/mu) / Γ((nu+2)/mu)
coeff_lambda = Γ((nu+1)/mu) / Γ((nu+2)/mu)
```

`particle_coeffs` (the struct these two routines fill) has fields `a_f, b_f, c_i, c_z`
(`mo_2mom_mcrph_types.f90:76-81`) — **not** `a_f, b_f, c_z, a_vel, b_vel` as an earlier
draft of this plan had it; `a_vel`/`b_vel` live on the particle-constants struct, not
the derived-coefficients struct. `cloud_coeffs` is a `particle_cloud_coeffs` (adds
`k_au, k_sc`, unused by the retained processes); `ice_coeffs`/`snow_coeffs`/
`graupel_coeffs`/`hail_coeffs` are `particle_sphere` (adds `coeff_alfa_n/q,
coeff_lambda` — also unused, per above).

**CCN coefficients (`aerosol_ccn`, `mo_2mom_mcrph_types.f90:131-140`)**: fields are
`Ncn0, Nmin, lsigs, R2, etas, wcb_min, z0, z1e` — **not** `a, b, c, d, e, f` as an
earlier draft had it (those letters belong to the unrelated `ccn_activation_hdcp2`
polynomial-fit constants). The 4D interpolation table itself (`tab`, populated once by
`get_otab`+`equi_table`) is a **separate object**, not a field on `ccn_coeffs` — port it
as its own lookup-table dataclass (grid vectors `x1..x4`, spacings `dx1..dx4`/`odx1..
odx4`, and the `ltable` array), built once at start-up exactly like Fortran's
`init_2mom_scheme_once` does, and passed alongside `ccn_coeffs` to the CCN stencil.

**Helper functions used across processes** (all in `mo_2mom_mcrph_processes.f90`,
pure scalar functions of a particle's static constants + local `q, n`):
`particle_meanmass(p,q,n) = clip(q/n, x_min, x_max)` (line 365),
`particle_diameter(p,x) = a_geo * x**b_geo` (line 378),
`particle_velocity(p,x) = a_vel * x**b_vel` (line 392), and
`diffusivity(T,p) = 8.7602e-5 * T**1.81 / p` (line 475).

---

## GT4Py Implementation Notes

See [`GT4Py_best_practices.md`](GT4Py_best_practices.md) for general field-operator/
program idioms, type aliases, and file-layout conventions shared with the rest of the
project — this section only covers what's specific to this scheme.

- **Module-level physical constants must be `enum.Enum(ta.wpfloat)` members, not
  plain Python floats, the moment they're referenced inside a
  `@gtx.field_operator`/`@gtx.program` body.** Confirmed the hard way in Phase 2
  (`stencils/unit_conversion.py`): plain floats (`RHO0 = 1.225`, etc.) referenced
  inside a field_operator body work fine under the embedded backend but fail to
  compile under `gtfn_cpu` with `EveValueError: Symbols {...} not found` — the
  compiled backend doesn't resolve arbitrary Python closures, only real
  parameters or recognized constant types. The fix (matching icon4py's own
  `MicrophysicsConstants(ta.wpfloat, enum.Enum)` pattern) is a plain drop-in:
  put the constants in an `enum.Enum` class, reference `MyConst.NAME` inside the
  field_operator body. One more wrinkle found the same way: an enum member
  can be referenced directly *inside* a field_operator's own body, but passing
  one as a call *argument* from a `@gtx.program` body into a nested
  field_operator call does not work (`TypeError: 'Attribute.value' must be
  Expr...`) — if a constant needs to vary between two calls in the same
  program (e.g. two different exponents), write two separate field_operators,
  each with its own constant baked into its body, rather than parameterizing
  one via an enum-valued argument threaded through the program. **Test every
  new stencil under `gtfn_cpu`, not just the default/embedded backend, before
  considering it done** — this class of bug is invisible under embedded and
  only surfaces once you compile for real.
- **`@gtx.program` bodies can't have ordinary Python statements at all, not
  just no-plain-constants** — found in Phase 3 writing `set_default_n`'s
  program (six field_operator calls sharing one domain): assigning
  `domain = {...}` once and reusing the local variable across the six calls
  fails with `UnsupportedPythonFeatureError: Unsupported Python syntax:
  'ast.Assign'`. A program body may only be a sequence of field_operator
  calls with fully inline arguments — no local variable assignment of any
  kind, not even for a literal dict passed straight through. Inline the
  `domain={...}` dict at every single call site instead, however repetitive
  that looks.
- **Give every module's private constants-enum a unique class name, not
  `_Const` everywhere.** Every stencil module so far (`unit_conversion.py`,
  `housekeeping.py`, `processes.py`) defined its own `class _Const(ta.wpfloat,
  enum.Enum)`. That's fine in isolation, but the moment one module's
  field_operator calls another's (as `ice_nucleation.py` does, calling
  `saturation.py`'s `e_es`/`e_ws`/`diffusivity`), compiling under `gtfn_cpu`
  fails: `NotImplementedError: Using closure vars with same name but different
  value across functions is not implemented yet. Collisions: '_Const'.`
  GT4Py resolves closure variables by name across the whole call graph being
  compiled together, so two same-named-but-different enums collide as soon as
  they're composed. Renamed every module's enum to a module-specific name
  (`_SaturationConst`, `_IceNucleationConst`, etc.) once this surfaced — do the
  same for any new stencil module from the start, don't wait to hit this.
  Also: functions called from inside a field_operator body must be imported
  and referenced by bare name (`from ...saturation import e_es`, then
  `e_es(temperature)`) — a module-qualified call (`saturation.e_es(...)`)
  fails with `DSLError: Functions can only be called directly.`
- **All retained processes are pointwise** (no neighbor or vertical-offset access), so
  the temp-field-reuse pattern (writing a stencil's output back into one of its own
  input fields via a `program`) is safe here. It would **not** be safe if sedimentation
  were ever added back (that needs `K`-offset reads), so don't copy this pattern
  reflexively into future work — `KOffset` isn't needed by anything in this plan and can
  be omitted from any starter templates.
- `vapor_dep_relaxation` involves four species (ice, snow, graupel, hail) and a rate
  split that sums across them, but it's still purely elementwise per level — each
  species' raw deposition rate and the joint `Xfac` rescaling are plain per-level
  arithmetic over multiple field arguments, no neighbor or vertical access. It fits a
  regular `@gtx.field_operator` fine; budget time for its length (5 fields in, 8 out),
  not for any control-flow difficulty.
- **`ccn_activation_sk_4d` is the one exception, and it's a real one, not a style
  choice.** Its 4D-table lookup indexes `tab%ltable` with indices computed at runtime
  from field values (`iu, ju, ku, lu` in `mo_2mom_mcrph_processes.f90:1770-1784`) — a
  data-dependent gather. GT4Py `field_operator`s don't have a general way to express
  "index a table by a value computed from another field," and reaching for
  `gtx.scan_operator` doesn't fix that (a scan still can't gather by an arbitrary
  runtime index). **Decision: implement this one routine — table construction *and*
  the per-step interpolation — as plain NumPy in the driver, not as a
  `field_operator`/`program`.** Extract the needed arrays out of their `Field`s, run
  the gather with ordinary NumPy indexing exactly as the Fortran does, and wrap the
  results back into `Field`s before the next compiled program call. This works
  identically for the embedded backend and for `gtfn_cpu` (or any other backend):
  compilation and backend restrictions only apply *inside*
  `@gtx.field_operator`/`@gtx.program`, and a driver-level Python function handing
  plain `Field`s to and from the surrounding compiled programs is exactly how a
  multi-step driver is already structured (see `SaturationAdjustment` in `icon4py`,
  which calls a sequence of separate compiled programs from ordinary Python). The
  tradeoff is that this one step won't benefit from gtfn compilation and would force a
  host/device copy on a GPU backend — irrelevant for a single CPU column now, and worth
  revisiting only as a later performance pass, not before the port is numerically
  correct end to end.
- The one-time coefficient setup (`setup_particle_coeffs`, `init_2mom_sedi_vel`,
  `get_otab`/`equi_table`) produces plain Python scalars/arrays, not GT4Py fields —
  compute it in ordinary Python/NumPy once at driver construction time and pass the
  results into stencils as scalar arguments (or, for the CCN LUT, keep it as the NumPy
  table object consumed directly by the NumPy CCN step above, not as stencil input).

## Icon4py Integration

`icon4py` is a peer workspace directory (`../../icon4py`), not a pip package — see its
`model/atmosphere/subgrid_scale_physics/microphysics/` for the real reference pattern
this plan follows (verified against the actual source, not assumed):

- `saturation_adjustment.py` already defines `SaturationAdjustmentConfig`,
  `MetricStateSaturationAdjustment`, and a `SaturationAdjustment` class with
  `_allocate_local_variables/_determine_horizontal_domains/_initialize_gt4py_programs`
  — reuse this directly for the two `satad_v_3d` calls rather than reimplementing satad.
  Note `input_properties()`/`output_properties()` on that class currently
  `raise NotImplementedError` even in `icon4py` itself — don't block on filling those in
  for a single-column driver.
- `microphysics_constants.py` defines `MicrophysicsConstants` as a
  `ta.wpfloat`-backed `enum.Enum` — put any new constants this scheme needs that aren't
  already in `icon4py.model.common.constants.PhysicsConstants` there, following that
  pattern, rather than re-deriving a separate constants module from scratch.
- `single_moment_six_class_gscp_graupel.py` / `stencils/graupel_stencils.py` /
  `stencils/microphysical_processes.py` are a **different** (single-moment) scheme —
  useful only as a packaging/file-layout reference (one `microphysical_processes.py` for
  shared stencils, as `GT4Py_best_practices.md` recommends), not as a source of physics
  to reuse; the actual formulas must come from this repo's Fortran, not from that file.
- This is a **single-column** driver (`column_driver.f90` reads two CSVs and runs one
  column, no MPI, no unstructured mesh). Don't build out `IconGrid`/horizontal-zone
  (`h_grid.Zone.NUDGING`/`LOCAL`) machinery for it — that's real infrastructure `icon4py`
  needs for the full model, but `mo_nwp_gscp_interface.f90`'s single-column
  `nwp_microphysics` has none of it. A `CellDim` of size 1 with a plain `KDim` covers the
  column; treat the full `MicrophysicsComponent`/CF-metadata protocol pattern as an
  optional later step for integrating with the rest of `icon4py`, not a prerequisite for
  getting the column right.

## Testing Strategy

Ground truth is `minimal_mcrph/fortran`'s own build: `make run` on
`example/hhl.csv`/`example/fields.csv` produces `example/output_fields.csv` (see
`column_driver.f90:1-20` for the exact CSV schema). Compare against that, not against
a reimplemented/hand-computed reference:

1. **Per-process unit tests**: for each stencil in `stencils/processes.py`, construct a
   small synthetic column (a handful of levels) with known inputs, and either
   (a) hand-derive the expected output from the formulas above, or (b) instrument the
   Fortran with an extra `WRITE` after the single call of interest and compare.
2. **Full-column integration test**: read `example/fields.csv`/`hhl.csv` the same way
   `column_driver.f90` does, run the GT4Py driver end to end (satad → prepare →
   processes → latent heat → post → satad), and compare every column against
   `example/output_fields.csv` at `rtol=1e-6` (looser than the Fortran-internal
   `atol=1e-13` used for pure recompilation checks, since GT4Py/NumPy and gfortran won't
   bit-match on `exp`/`log`/`GAMMA`).
3. **Process-ordering regression test**: verify each stencil call reads the *previous*
   stencil's output (not the pre-chain input) — e.g. assert that `cloud_freeze` sees the
   `cloud%q` already updated by CCN activation, not the CSV's raw `qc` column. This is
   what actually enforces the call-graph ordering in the "Process Call Graph" section
   above, not a docstring comment.
4. Keep the number-concentration clipping calls (steps 3, 6, 9 in the call graph) as
   part of what's tested end-to-end — they change results whenever a hydrometeor's mean
   mass would otherwise leave `[x_min, x_max]`, which is common enough in the example
   column to matter for the `rtol=1e-6` comparison.

## Implementation Roadmap

1. **Foundation**: `particles.py` (static per-hydrometeor constants), `config.py`,
   coefficient setup (`setup_particle_coeffs`, `init_2mom_sedi_vel`, the CCN 4D table) —
   all pure Python/NumPy, computed once, no GT4Py yet.
2. **Unit-conversion stencils**: `prepare`/`post` density transforms + the latent-heat
   temperature update — these are simple, pointwise, and needed before any process
   stencil's output means anything in the CSV's mixing-ratio units.
3. **Process stencils**, in this order (easiest/most isolated first, but all are needed
   before the integration test can pass):
   - `set_default_n` and the three clipping stencils (no physics, good GT4Py warm-up)
   - `cloud_freeze`, `ice_melting` (simple per-level `where` chains)
   - `ccn_activation_sk_4d` — implement as a plain NumPy driver step (table gather, see
     "Process Interfaces" and "GT4Py Implementation Notes" above), not a
     `field_operator`; this is the real target, get its numerics right against the CSV
     reference. Use `ccn_activation_hdcp2` only as an optional earlier smoke test to
     shake out GT4Py idioms on the rest of the pipeline, never as a substitute milestone.
   - `ice_nucleation_homhet` (dispatches to `ice_nucleation_het_inas` for the default
     config, `nuc_i_typ=1`, plus homogeneous KHL06 — the largest routine, but ordinary
     elementwise `field_operator` code; budget time for length/transcription risk, not
     for a GT4Py-fit problem)
   - `vapor_dep_relaxation` (multi-species elementwise `field_operator`, see above)
4. **Driver**: wire the above into `stripped_mcrph_driver.py` matching
   `two_moment_mcrph`'s prepare → `clouds_twomoment` → latent heat → post structure
   (with the CCN step as the one plain-Python link in that otherwise-compiled chain),
   then `satad → driver → satad` matching `nwp_microphysics`.
5. **Validate** against `example/output_fields.csv` per the testing strategy above. When
   the whole-column comparison fails, add temporary `WRITE`s to
   `mo_2mom_mcrph_main.f90::clouds_twomoment` to dump `cloud/rain/ice/snow/graupel/hail
   %q,%n` and `atmo%qv` after each of the 5 process calls, and compare stage by stage
   instead of debugging blind against only the final CSV. Note the existing
   `check(...)` calls at those same call sites (guarded by `ischeck`) do **not** do
   this — they only assert `q >= -1e-12` and abort otherwise (`mo_2mom_mcrph_main.f90:
   790-830`), so they catch a NaN/negative-value bug but give no comparison data; the
   dumping has to be added. Revert these `WRITE`s once done — they're a debugging aid,
   not part of the reduced scheme.
6. Once the full column matches: revisit whether `ccn_activation_sk_4d` is worth
   expressing as a compiled stencil (e.g. unrolling its small, fixed `r2 × lsigs` grid
   into `where` cascades) — a performance question for a full 3D grid / GPU backend,
   deliberately deferred past initial correctness.

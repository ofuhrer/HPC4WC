# Plan: `minimal_mcrph/fortran` — a stripped 2-moment microphysics

## Goal

Build an **intermediate** single-column sandbox that sits between
[`satad_only/fortran`](../satad_only/fortran) (saturation adjustment alone) and
the full scheme in [`fortran/`](../fortran). It runs only the microphysical
processes that need **neither gamma-function lookup tables nor a vertical
(sedimentation) solver**, so it stays tractable to port to GT4Py.

The retained call sequence, per the task:

1. saturation adjustment (`satad_v_3d`)
2. microphysics — a **reduced** `clouds_twomoment`:
   a. CCN activation (`ccn_activation_*`)
   b. INP nucleation (`ice_nucleation_homhet`)
   c. cloud-droplet freezing (`cloud_freeze`)
   d. ice depositional growth (`vapor_dep_relaxation`)
   e. ice-crystal melting (`ice_melting`)
3. saturation adjustment again (`satad_v_3d`)

Everything else in the full `clouds_twomoment` — ice/snow/graupel/hail
collisions, riming, `graupel_hail_conv_wet_gamlook`, `rain_freeze_gamlook`,
snow/graupel/hail melting, evaporation, warm-rain autoconversion/accretion,
rain evaporation — is **removed**, and so is all sedimentation.
`cloud_mass_growth` is **also removed** (decision below): the two satad calls
already handle cloud-droplet condensation/evaporation, so the retained block is
exactly a → e with nothing interleaved.

## Why this eliminates the hard parts

- **Gamma lookups.** In the full one-time init
  ([`init_2mom_scheme_once`, mo_2mom_mcrph_main.f90:877](../fortran/mo_2mom_mcrph_main.f90#L877))
  every `incgfct_lower_lookupcreate` call feeds *only* `rain_freeze_gamlook`,
  `graupel_hail_conv_wet_gamlook`, and the shedding riming routines — all of
  which we drop. So removing those process calls lets us delete the gamma-table
  setup entirely.
- **netCDF / wet-growth `dmin` tables.** `init_dmin_wg_gr_ltab_equi` (the only
  netCDF I/O) feeds the wet-growth conversion routines we drop. netCDF is
  `USE`d by `mo_2mom_mcrph_dmin_wetgrowth.f90` and by the `dmin_wg_gr_ltab_equi`
  section of `mo_2mom_mcrph_util.f90`. Removing those lets us drop the netCDF
  build dependency, exactly as `satad_only` did.
- **Vertical solver.** `sedimentation_explicit()` /
  `clouds_twomoment_implicit()` in
  [`mo_2mom_mcrph_driver.f90`](../fortran/mo_2mom_mcrph_driver.f90) are the only
  vertical coupling; the retained processes are all per-level. We keep the
  driver's per-level latent-heat temperature update but delete both
  sedimentation paths.

## Coefficient dependencies of the 5 retained routines (must survive init)

- `ccn_activation_*` → `ccn_coeffs` / Hande tables (set in `two_moment_mcrph_init`).
- `ice_nucleation_homhet` → `in_coeffs` and ice-nucleation config.
- `cloud_freeze` → `cloud_coeffs` (from `setup_particle_coeffs`), `qnc_const`.
- `vapor_dep_relaxation` → `ice_coeffs`, `snow_coeffs`, `graupel_coeffs`,
  `hail_coeffs` from `setup_particle_coeffs` + `init_2mom_sedi_vel` (these use
  the **complete** gamma function `gfct`, an elemental call — *not* a lookup
  table — so they stay).
- `ice_melting` → no special coeffs.

None of these need incomplete-gamma lookups or the wet-growth tables, which is
what makes the reduction clean.

## Approach

**Minimal driver + minimal process sequence + a stripped `processes` module.**
Mirror `satad_only`'s layout: copy the scheme's source into
`minimal_mcrph/fortran`, change the orchestration, and — per the request —
replace `mo_2mom_mcrph_processes.f90` with a copy containing **only the
subroutines the reduced call tree actually needs** (the dependency closure
below). `mo_2mom_mcrph_main.f90`, `mo_2mom_mcrph_util.f90`, config and types
stay largely intact.

1. **`clouds_twomoment`** (in `mo_2mom_mcrph_main.f90`): cut its body down to
   set-up bookkeeping (`set_default_n`, size clamps) plus the 5 retained calls.
   Keep the `nuc_c_typ` CCN-activation dispatch and the `iicephase==1` guard.
   **Do not** call `cloud_mass_growth` in either the `lexpl_supersat` or the
   `.not.lexpl_supersat` position — both calls are deleted.
2. **`init_2mom_scheme_once`**: delete the `incgfct_lower_lookupcreate` blocks,
   the shedding/collision coefficient setups, and the sticking-efficiency
   tables that only the dropped routines use. Keep `setup_particle_coeffs`,
   `init_2mom_sedi_vel`, rain/`mu`-relation, and CCN/IN setup.
3. **`two_moment_mcrph` / `two_moment_mcrph_init`** (driver): remove
   `init_dmin_wg_gr_ltab_equi` and both sedimentation solvers; keep prepare →
   `clouds_twomoment` → latent-heat temperature update → post.
4. **`mo_nwp_gscp_interface.f90`**: unchanged — it already does satad → scheme →
   satad, which is precisely the target flow.
5. **Strip `mo_2mom_mcrph_processes.f90`**: keep the module header (USE list,
   module variables/parameters, PUBLIC declarations) and only the routines in
   the keep-list below; delete the other ~80 routines. Update the module's
   `PUBLIC` list accordingly. (The header's module-level parameter/lookup arrays
   used by the retained nucleation routines — INAS/Philips coefficient tables —
   must be kept; verify none of them are populated by a deleted setup routine.)
6. **Drop netCDF**: remove `mo_2mom_mcrph_dmin_wetgrowth.f90` and the
   netCDF-using `dmin` lookup section of `mo_2mom_mcrph_util.f90`; strip netCDF
   from the Makefile (copy `satad_only`'s netCDF-free Makefile as the base and
   extend its `SRCS` list).

The retained physics stays **bit-identical** to the full scheme, but the
process module now exposes only the small, honest call tree the GT4Py port has
to reproduce.

### Keep-list for the stripped `processes` module

Computed as the transitive call closure of the 5 process routines +
`set_default_n` + the two init helpers, over routine bodies with comments and
string literals stripped (24 routines; `equi_table` is nested inside
`get_otab`, so 23 top-level extraction units):

| group | routines |
|---|---|
| particle/util helpers | `particle_meanmass`, `particle_diameter`, `particle_velocity`, `vent_coeff_a`, `vent_coeff_b`, `moment_gamma`, `diffusivity` |
| init helpers (kept in trimmed `init_2mom_scheme_once`) | `setup_particle_coeffs`, `init_2mom_sedi_vel` |
| bookkeeping | `set_default_n` |
| CCN activation | `ccn_activation_hdcp2`, `ccn_activation_sk_4d` → `get_otab` (→ contained `equi_table`) |
| INP nucleation | `ice_nucleation_homhet` → `ice_nucleation_het_inas` (→ `het_icenuc_inas_depo`), `ice_nucleation_het_philips`, `ice_nucleation_agi_dm95`, `ice_nucleation_agi_m16` |
| cloud freezing | `cloud_freeze` |
| ice deposition | `vapor_dep_relaxation` → `vapor_deposition_generic` |
| ice melting | `ice_melting` |

Notes:
- `ice_nucleation_het_hdcp2` is **not** kept — `ice_nucleation_homhet` only
  references it in a commented-out call, so it is dead code.
- `ccn_activation_sk` (and its contained `lookuptable`, `ip_ndrop_ncn`,
  `ip_ndrop_wcb`) is **not** kept — `clouds_twomoment` dispatches only to
  `ccn_activation_hdcp2` / `ccn_activation_sk_4d`.
- None of the retained routines are type-bound procedures (the `particle` types
  declare no `PROCEDURE => …` bindings), so dropping the others cannot break a
  type definition.
- The exact keep-set is reproducible via `scratchpad/closure2.py`.

## Deliverable layout (mirrors `satad_only/fortran`)

```
minimal_mcrph/
  PLAN.md                      this file
  fortran/
    README.md                  adapted from satad_only + fortran READMEs
    Makefile                   netCDF-free, hand-ordered SRCS
    column_driver.f90          CSV in -> init -> nwp_microphysics -> CSV out
    mo_nwp_gscp_interface.f90  satad -> reduced scheme -> satad (unchanged)
    mo_2mom_mcrph_driver.f90   no sedimentation, no dmin init
    mo_2mom_mcrph_main.f90     reduced clouds_twomoment + trimmed init
    mo_2mom_mcrph_processes.f90
    mo_2mom_mcrph_util.f90     netCDF/dmin section removed
    mo_2mom_mcrph_config*.f90
    mo_2mom_mcrph_types.f90
    mo_2mom_prepare.f90
    mo_satad.f90
    dependencies/*.f90         mo_kind, mo_exception, mo_physical_constants,
                               mo_lookup_tables_constants, mo_timer, mo_reff_types
    example/                   hhl.csv, fields.csv (reuse full scheme's inputs)
    agents_plan/               copy of this plan (per repo convention)
```

## Steps

1. Scaffold `minimal_mcrph/fortran` by copying the needed sources from
   `fortran/`; take the Makefile from `satad_only/fortran` and extend `SRCS`.
2. Produce the stripped `mo_2mom_mcrph_processes.f90` (header + keep-list
   routines only); update its `PUBLIC` list.
3. Trim `clouds_twomoment` (drop `cloud_mass_growth` + everything after
   `ice_melting`) and `init_2mom_scheme_once` in `mo_2mom_mcrph_main.f90`.
4. Trim the driver (`two_moment_mcrph`, `two_moment_mcrph_init`): drop
   sedimentation + `dmin` init.
5. Remove the netCDF `dmin` code from `mo_2mom_mcrph_util.f90`; drop
   `mo_2mom_mcrph_dmin_wetgrowth.f90`; confirm no remaining `USE netcdf`.
6. `make check`, then `make`, then `make run` on the example column; confirm it
   builds with no netCDF and produces sensible output.
7. Write the `README.md`; copy this plan into `agents_plan/`.
8. Open a PR (per repo rules: one per completed step / plan).

## Decisions (resolved)

1. **`cloud_mass_growth` — removed.** The two satad calls already handle
   cloud-droplet condensation/evaporation; both call sites are deleted.
2. **Stripped `processes` module — yes.** Only the keep-list routines are
   carried over (see above).

## Decisions taken during implementation

1. **netCDF removed.** `mo_2mom_mcrph_dmin_wetgrowth.f90` is deleted and the
   netCDF/dmin-wetgrowth section of `mo_2mom_mcrph_util.f90` is excised; the
   Makefile no longer probes for `nf-config`. No `.nc` files are produced.
2. **Incomplete-gamma lookup machinery removed** (not just skipped). The
   `gamlookuptable` type and the whole `incgfct_*` family are gone from
   `util.f90`, and the `gamlookuptable` module variables/`USE`s are gone from
   `main.f90`. `util.f90`: 2033 → ~520 lines; `processes.f90`: 7290 → ~2118.
3. **LWF melting includes removed.** `include/hailcoeffs.incf` and
   `include/grplcoeffs.incf` (LWF melting-scheme coefficients, used only by the
   dropped `particle_melting_lwf`/`prepare_melting_lwf`) are deleted, along with
   their module-level coefficient arrays. `include/phillips_nucleation_2010.incf`
   stays (feeds the retained Phillips/INAS ice nucleation).
4. **Solver path forced to explicit.** The driver's semi-implicit solver and
   `sedimentation_explicit`/`clouds_twomoment_implicit` are removed; the explicit
   body (save-old → `clouds_twomoment` → latent-heat temperature update) always
   runs and `cfg_params%i2mom_solver` is ignored (with no sedimentation the two
   solvers are equivalent).
5. **`ischeck`/`isdebug` diagnostics** referencing dropped hydrometeors — the
   remaining `check(...)` calls are kept; they're harmless and aid debugging.

## Status: implemented and building

`make` / `make check` succeed with no netCDF; `make run` on the bundled example
column produces sensible saturation-adjustment behaviour (supersaturated layers
condense vapour→cloud and warm; sub-saturated layers pass through) and zero
surface precipitation, as expected for a scheme with no precipitation-size
particles.
```

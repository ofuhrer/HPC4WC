# Convert the 2-moment microphysics sandbox to a single-column model

## Context

This sandbox (`/Users/bbuchenau/cloud_code/hpc4wc_microphysics_sandbox/`) holds a copy of
ICON's 2-moment bulk microphysics scheme, previously trimmed to only the `inwp_gscp==5`
code path. ICON vectorizes physics over three nested index dimensions: `jb` (a block of
grid columns, chosen for cache/vector-length reasons), `jc` (a horizontal grid point within
a block), and `jg` (a nested-domain index used to look up per-domain config). The user wants
all of this stripped away so the code operates as a genuine single-column model: only the
vertical-level loop survives.

Exploration (3 Explore agents + 1 Plan agent, full-file reads) confirms:
- **`jb`/`jc`/`jg` only appear in `mo_nwp_gscp_interface.f90`.** Everywhere else, the
  point dimension shows up as a generic `(i,k)` or `(ii,kk)` pair over `its:ite`/`istart:iend`
  (or packed into `ik_slice(4)` in two files), with no block/domain concept at all — those were
  already resolved one layer up, in the interface.
- Three files are pure scalar/lookup-table code and need **no changes**:
  `mo_2mom_mcrph_util.f90`, `mo_2mom_mcrph_config_default.f90`, `mo_2mom_mcrph_dmin_wetgrowth.f90`.
- Only `mo_kind.f90` and `mo_physical_constants.f90` (leaf dependencies) have been vendored into
  the sandbox so far — the heavier ICON grid/patch/config modules (`mo_model_domain`,
  `mo_nonhydro_types`, `atm_phy_nwp_config`, etc.) have not, and live only as reference copies
  under `other/` (leave that directory untouched — it's original ICON source for lookup, not
  part of the buildable sandbox). This confirms the interface should become a standalone
  column driver rather than gain a pile of new stub dependencies.
- Pre-existing, out-of-scope gap: `mo_2mom_mcrph_driver.f90`/`mo_2mom_mcrph_processes.f90`/
  `mo_2mom_mcrph_config_default.f90` all `USE mo_2mom_mcrph_config, ONLY: t_cfg_2mom`, but that
  module doesn't exist in the sandbox. Not touching this — it predates this task and is
  unrelated to loop removal (t_cfg_2mom is pure scalars; if the user later wants to build this,
  it's a one-file vendor-in, not a refactor).

No Fortran compiler is available in this environment, so verification is structural
(grep/consistency checks), not a real build — noted at the end.

## Confirmed decisions

1. **`nwp_microphysics` becomes a standalone column driver**: plain array/scalar arguments,
   no ICON derived types (`t_patch`, `t_nh_prog`, `t_nh_diag`, `t_nwp_phy_diag`,
   `t_nwp_phy_tend`, `t_external_data`, `atm_phy_nwp_config`), no ICON infra `USE`s
   (`mo_model_domain`, `mo_nonhydro_types`, `mo_nwp_phy_types`, `mo_ext_data_types`,
   `mo_run_config`, `mo_loopindices`, `mo_impl_constants*`, `mo_nwp_diagnosis`,
   `mo_grid_config`, `mo_timer`, `mo_fortran_tools`, `mo_parallel_config`).
2. **Do the whole conversion in one pass**, in dependency order, across every file that needs it.

## Approach

### The pivot: `mo_2mom_mcrph_types.f90`
`TYPE atmosphere` (`w,p,t,rho,qv,zh,tke`) and `TYPE particle`/`particle_frozen`/`particle_lwf`
(`n,q,rho_v`, and `l` for LWF) currently hold `POINTER, DIMENSION(:,:)` fields (point × level).
Change all of these to `POINTER, DIMENSION(:)` (level only). Every other file's array shape
flows from this change, so it must land first. Leave `particle_coeffs`/`particle_sphere`/…
(scalar per-species coefficients) and the `lookupt_1D/2D/4D` table types (parameter-space axes,
unrelated to the model grid) untouched.

### The mechanical (i,k) → (k) transform, applied everywhere else
Representative before/after (`autoconversionSB` in `mo_2mom_mcrph_processes.f90`, but this
pattern repeats near-identically across ~100 subroutines there, ~5 loop nests in
`mo_2mom_mcrph_main.f90`, and 3 in `mo_2mom_prepare.f90`):

```fortran
! before
SUBROUTINE autoconversionSB(ik_slice,dt,atmo,cloud_coeffs,cloud_in,rain)
  INTEGER, INTENT(in) :: ik_slice(4)
  ...
  istart = ik_slice(1); iend = ik_slice(2); kstart = ik_slice(3); kend = ik_slice(4)
  !$ACC LOOP GANG VECTOR COLLAPSE(2)
  DO k = kstart,kend
    DO i = istart,iend
      q_c = cloud%q(i,k)
      ...
      rain%n(i,k) = rain%n(i,k) + au * x_s_i
    END DO
  END DO

! after
SUBROUTINE autoconversionSB(kstart,kend,dt,atmo,cloud_coeffs,cloud_in,rain)
  INTEGER, INTENT(in) :: kstart, kend
  ...
  DO k = kstart,kend
    q_c = cloud%q(k)
    ...
    rain%n(k) = rain%n(k) + au * x_s_i
  END DO
```

Rules to apply consistently:
- Drop `ik_slice(4)` in favor of two plain scalar arguments `kstart, kend` everywhere (including
  `mo_2mom_mcrph_main.f90`'s `clouds_twomoment` and its ~30 downstream calls into
  `mo_2mom_mcrph_processes.f90`). `mo_2mom_prepare.f90` already uses bare `its,ite,kts,kte` —
  there, just delete `its, ite` and keep `kts, kte` as-is (less diff). Don't introduce a
  2-element `k_slice(2)`; it buys nothing over two named scalars and only adds unpacking
  boilerplate back in.
- Delete the outer point-loop and its `END DO`; de-indent the body; every `foo(i,k[±1])` →
  `foo(k[±1])`.
- Drop `!$ACC`/`!$OMP` directives wrapping the removed loops (no more horizontal axis to
  vectorize over). `mo_2mom_prepare.f90`'s `__acc_attach(...)` pointer-attach macros go too.
- `MINVAL`/`MAXVAL`/`ANY` reductions over the vanished point range collapse to the bare scalar
  (e.g. `MINVAL(cloud%q(:,k))` → `cloud%q(k)`).
- Keep explicit `kstart,kend` (or `kts,kte`) scalar bound arguments alongside assumed-shape
  `DIMENSION(:)` field arguments — do **not** hardcode `DIMENSION(nlev)` with implicit `1:nlev`
  loops. `kstart` is frequently `> 1` (moist-physics start level excludes upper stratosphere),
  so the explicit-bounds idiom is physically correct, not just convenient.
- Audit `SIZE(x, 1)`/`SIZE(x, 2)` calls on anything whose rank changed 2→1 (e.g.
  `mo_2mom_mcrph_main.f90`'s local `dep_rate_ice(size(cloud%n,1),size(cloud%n,2))` →
  `dep_rate_ice(SIZE(cloud%n))`; `mo_2mom_mcrph_processes.f90`'s
  `SIZE(atmo%rho,dim=2)` → `SIZE(atmo%rho)`).

### Special-case: genuinely sequential level-marching sedimentation (highest risk — do not flatten this state)
Two solvers, both reachable in practice (default config has `i2mom_solver=1`, so the implicit
path is not a rare branch):

- **`clouds_twomoment_implicit`** (`mo_2mom_mcrph_driver.f90`, default path): its per-species
  flux/sum/impl state (`qr_flux_now, qr_sum, qr_impl, vr_sedq_now, ...`) is already per-point,
  not per-level — it becomes **plain scalars**, declared once outside `DO k=kts+1,kte` and
  overwritten each iteration, exactly as a hand-written column model would do it. Convert
  `implicit_core`/`implicit_time` (module-level, currently `DIMENSION(:)` over `its:ite` for a
  fixed k-slice) to plain scalar arguments. Preserve `DO k = kts+1, kte` exactly (starts one
  level below the top by design — don't "simplify" to `kts`).
- **`sedi_icon_core`/`sedi_icon_box_core`** (+ `_lwf` twins, `mo_2mom_mcrph_processes.f90`,
  used when `i2mom_solver=0`): in `sedi_icon_core`, `s_nv`/`s_qv` are per-point-per-iteration
  scalars already — fine to scalarize fully. **In `sedi_icon_box_core`, `s_nv`/`s_qv` are
  `DIMENSION(its:ite,kts:kte)` and are written *forward* into future levels
  (`s_nv(i,k+kk) = s_nv(i,k+kk) + ...`) across outer-loop iterations** — these must stay
  full `DIMENSION(kts:kte)` 1D arrays over k after the point axis is dropped; only remove the
  `i`/point dimension, never the `k` dimension here. This is the single most important thing to
  get right in the whole refactor — a naive blanket "(i,k)→(k)" pass would incorrectly flatten
  this cross-iteration state to a scalar and silently break sedimentation.
- `sedi_vel_rain`/`sedi_vel_sphere`/`sedi_vel_lwf` (1D-over-point-only, called once per level by
  the caller) convert to plain scalar in/out.

### Special-case: gather/scatter vectorization tricks — simplify away, don't preserve
Two places use manual index-compaction (`PACK`-style, a CPU/NEC vectorization trick, not
physics) that becomes pointless overhead with one column and should be replaced by a direct
per-level `IF`, not mechanically preserved:
- `ice_nucleation_het_inas` (`mo_2mom_mcrph_processes.f90`): drop the `ii(j),kk(j)` index-list
  build; use a plain `DO k=kstart,kend` with an inline `IF`.
- `satad_v_3D` (`mo_satad.f90`): drop the `iwrk/kwrk/twork/tworkold` compaction; model the
  replacement on `satad_v_3D_gpu`, which is already point-independent per level (a per-level
  `DO count=1,maxiter` Newton loop) — just drop its point loop and keep that structure.

### `mo_nwp_gscp_interface.f90` rewrite
Drop: both block loops, the nested point loops, `l_limited_area`/`l_nest_other_micro`/
`grf_bdywidth_c`/`get_indices_c`/`i_rlstart`/`i_rlend`/`i_startblk`/`i_endblk`/`i_startidx`/
`i_endidx` (nest/blocking machinery), all ICON derived-type arguments, `atm_phy_nwp_config(jg)%…`
and `kstart_moist(jg)` (→ plain scalar `kstart` argument), the nested-domain boundary-density
block (lines ~140-183 of the current file — no meaning without nesting, delete along with its
now-unused `mo_2mom_mcrph_util` import), `lavail_tke`/`atm_phy_nwp_config(jg)%cfg_2mom%lturb_enhc`
(just forward whatever `tke` pointer the caller passes, associated or not — `prepare_twomoment`
already handles a null pointer), and the `nwp_diag_output_minmax_micro` diagnostic calls (ICON
multi-column diagnostics, no sandbox equivalent).

Keep, as new plain arguments: full field arrays (`tk, qv, qc, qnc, qr, qnr, qi, qni, qs, qns,
qg, qng, qh, qnh`, optional `ssat, nccn, ninpot, ninagi`), layer geometry (`dz, hhl, rho, pres,
w`, optional `tke` pointer), `qrsflux`, precip rates (now plain scalars `prec_r, prec_i, prec_s,
prec_g, prec_h`, not `nproma`-sized), `dt`, `lsatad`, `nlev`/`kstart`, and the physics-switch
scalars that used to come from `atm_phy_nwp_config(jg)`: `ithermo_water, ice_type
(i2mom_icenucleation), luse_agi, iagi_param, lexpl_supersat, msg_level, l_cv`.

Must preserve the exact call sequence added in the prior conversation turn: satad →
`two_moment_mcrph` → satad, plus the surface-precip-rate accumulation (now plain scalar
arithmetic, no `jc` loop).

`lcompute_tt_lheat`/`tt_lheat`: make it an **optional** argument pair (`OPTIONAL` logical +
`OPTIONAL` plain `(nlev)` array, caller-owned) rather than dropping it outright — it's an
ICON-LHN-specific diagnostic with no sandbox type to back it, but the subtract-before/add-after
bookkeeping itself is generic and harmless to keep available, off by default.

## Execution order

1. `mo_2mom_mcrph_types.f90` (pivot — must land first)
2. `mo_2mom_prepare.f90`
3. `mo_2mom_mcrph_processes.f90` (bulk of the work; do the sedimentation and gather/scatter
   special-cases carefully, everything else is the mechanical transform)
4. `mo_2mom_mcrph_main.f90` (also fix the `SIZE(cloud%n,1/2)` local-array-declaration bug this
   surfaces)
5. `mo_2mom_mcrph_driver.f90` (`two_moment_mcrph`, `clouds_twomoment_implicit`,
   `sedimentation_explicit`, `implicit_core`, `implicit_time`)
6. `mo_satad.f90`
7. `mo_nwp_gscp_interface.f90`

No changes: `mo_2mom_mcrph_util.f90`, `mo_2mom_mcrph_config_default.f90`,
`mo_2mom_mcrph_dmin_wetgrowth.f90`.

## Verification (no compiler available in this environment)

- Grep for `ik_slice`, `its`/`ite`/`istart`/`iend` (as whole words), `jc`/`jb`/`jg` across all
  changed files — should return zero hits (spot-check any surprise hit for an unrelated
  variable reusing the same letter).
- Grep remaining `DIMENSION(:,:)`/`POINTER, DIMENSION(:,:)` in the changed files — every hit
  should be a genuine lookup-table/coefficient array (parameter-space axes), not a missed
  point/level field.
- Explicitly re-check `s_nv`/`s_qv` in `sedi_icon_box_core` are still `DIMENSION(kts:kte)`, not
  scalars — the highest-value single check in this refactor.
- Re-check `clouds_twomoment_implicit`'s loop still reads `DO k = kts+1, kte`.
- DO/ENDDO and IF/ENDIF balance check per file after loop deletions (same approach used earlier
  in this session for `mo_nwp_gscp_interface.f90`).
- For every changed subroutine signature, grep all call sites (same file and cross-file) and
  diff argument count/order against the new declaration — particularly `sedi_icon_core`/
  `sedi_icon_box_core` (called from `sedi_icon_rain/sphere/sphere_lwf`) and `implicit_core`/
  `implicit_time` (~12 call sites each inside `clouds_twomoment_implicit`).
- Final end-to-end read of the rewritten `mo_nwp_gscp_interface.f90` against the "must remain"
  list (satad → two_moment_mcrph → satad ordering; precip accumulation; optional LHN hook).
- No local compiler is available, so an actual build/run in the user's real ICON/HPC
  environment remains the ground-truth check — flag this limitation when reporting back.

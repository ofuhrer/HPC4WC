# Design plan — porting ICON `satad` (saturation adjustment) to GT4Py

*Status: Step 3 design plan, awaiting review (Step 4) before any code is written.*
*Target Fortran: `satad_only/fortran/mo_satad.f90` (`satad_v_3D` / `satad_v_3D_gpu`).*
*Constraint honoured throughout: the `icon4py/` reference port is held out and has not been opened.*

---

## Key DSL constraint driving the design

GT4Py's field-operator frontend **forbids `for` and `while` loops** (they are on the
unsupported-feature list; the error message points to `scan_operator` for sequential
dependencies). There is therefore no static `for`-unrolling to rely on — the Newton
iteration is unrolled manually. The math builtins `exp` and `maximum` both exist, so the
active (Tetens) code path is fully expressible.

---

## 1. File structure

```
satad_only/gt4py/
  satad_gt4py.py        # the port: dimensions, constants, field_operators, program,
                        # plus a numpy driver wrapper
  test_satad_gt4py.py   # standalone self-check (see section 7)
  README.md             # how to run/test without this conversation
```

Self-contained, sits alongside `satad_only/fortran/`. Does not touch `icon4py/`.

## 2. Dimensions & public interface

Fields are laid out over **`CellDim x KDim`** (standard ICON layout; a single column is
just `num_cells = 1`). Everything is elementwise — `K` carries no coupling, consistent
with the Fortran.

```python
CellDim = gtx.Dimension("Cell")
KDim    = gtx.Dimension("K")
wpfloat = float64
```

Entry point (a `@program`), mirroring `satad_v_3D`:

```
satad(
    te:     Field[Dims[CellDim, KDim], float64],   # temperature       [K]      IN/OUT
    qve:    Field[Dims[CellDim, KDim], float64],   # specific humidity [kg/kg]  IN/OUT
    qce:    Field[Dims[CellDim, KDim], float64],   # cloud water       [kg/kg]  IN/OUT
    rhotot: Field[Dims[CellDim, KDim], float64],   # total density     [kg/m^3] IN
    tol:    float64,                                # temp accuracy     [K]
    out=(te, qve, qce),                             # write-back
)
```

`maxiter` is **not** a runtime argument (see section 5) — it is fixed at 10 and
documented. A thin numpy wrapper `satad_numpy(rho, tk, qv, qc, tol=1e-3)` is also
provided so an evaluator can call the port on plain arrays / the CSV without knowing
GT4Py.

## 3. Constants (transcribed exactly, `float64`)

| Symbol | Value / formula | Notes |
|---|---|---|
| `rd`     | `287.04`               | gas constant, dry air |
| `rv`     | `461.51`               | gas constant, water vapour |
| `cpd`    | `1004.64`              | c_p dry air |
| `cvd`    | `cpd - rd`             | = 717.60 |
| `tmelt`  | `273.15`               | b3 |
| `alv`    | `2.5008e6`             | latent heat of vaporisation (lwd) |
| `cp_v`   | `1850.0`               | **local satad PARAMETER — NOT `cpv=1869.46`** |
| `clw`    | `(3.1733 + 1.0) * cpd` | specific heat of liquid water (cl) |
| `c1es`   | `610.78`               | Tetens b1 |
| `c3les`  | `17.269`               | Tetens b2w |
| `c4les`  | `35.86`                | Tetens b4w |
| `c5les`  | `c3les * (tmelt - c4les)` | Tetens b234w |
| `zqwmin` | `1.0e-20`              | floor on adjusted qc |

## 4. Helper functions (as `@field_operator`s, active `ipsat=1` path only)

```
latent_heat_vaporization(T) = alv + (cp_v - clw) * (T - tmelt) - rv * T
sat_pres_water(T)           = c1es * exp( c3les * (T - tmelt) / (T - c4les) )
qsat_rho(T, rho)            = sat_pres_water(T) / (rho * rv * T)
dqsatdT_rho(qsat, T)        = ( c5les / (T - c4les)**2 - 1.0 / T ) * qsat
```

The Murphy–Koop formulation (`ipsat=2`) is dead code and will **not** be ported.

## 5. Control flow in the DSL (the tricky bits)

**Branching becomes `where`.** Compute once, from the *input* `te`:

```
lwdocvd   = latent_heat_vaporization(te) / cvd
Ttest     = te - lwdocvd * qce
qtest     = qsat_rho(Ttest, rho)
qw        = qve + qce
needs_iter = qw > qtest
```

**Newton iteration becomes a manual unroll.** A one-step helper field_operator:

```
_newton_step(twork, tworkold, te, qve, lwdocvd, rho, tol, needs_iter):
    active    = needs_iter AND |twork - tworkold| > tol   # frozen once converged
    qwd       = qsat_rho(twork, rho)
    dqwd      = dqsatdT_rho(qwd, twork)
    fT        = twork - te + lwdocvd * (qwd - qve)
    dfT       = 1.0 + lwdocvd * dqwd
    twork_new = where(active, twork - fT / dfT, twork)
    return twork_new, twork          # old twork becomes the new tworkold
```

Called **10 times in sequence** (init `twork = te`, `tworkold = te + 10` to force the
first step). This reproduces the **GPU variant's** semantics exactly (fixed count,
per-step mask) and is numerically identical to the CPU `while`-loop, because once
`|Δtwork| <= tol` the update is masked off.

**Closure + branch selection.**

```
qwa     = qsat_rho(twork, rho)
te_B    = twork
qce_B   = maximum(qce + qve - qwa, zqwmin)
qve_B   = qwa
# branch A (all cloud evaporates, still sub-saturated): qve = qw, qce = 0, te = Ttest
te_out  = where(needs_iter, te_B,  Ttest)
qve_out = where(needs_iter, qve_B, qw)
qce_out = where(needs_iter, qce_B, 0.0)
```

## 6. Backend

Default **embedded** (`backend=None`, pure numpy execution) — no C++ toolchain needed, so
the port runs anywhere. `gtfn_cpu` is mentioned in the README as an optional compiled
backend.

## 7. Verification

- Build/run the Fortran driver (`make run` in `satad_only/fortran`) to produce
  `example/output_fields.csv`; run the GT4Py port on the same `fields.csv`; assert
  `np.allclose` on `tk / qv / qc` (expected agreement ~1e-10; not bit-identical, due to
  floating-point op ordering in `exp` / division).
- Hand-built edge cases: a dry sub-saturated level (unchanged), a level that fully
  evaporates (branch A), a supersaturated level (branch B).
- Delivered as a runnable `test_satad_gt4py.py`.

## Decisions resolved

- **Layout:** `CellDim x KDim` (a column = 1 cell).
- **Iteration:** manual 10x unroll (no loops allowed in the DSL); `maxiter` fixed at 10 to
  match the reference driver.
- **Backend:** embedded by default.

## Open items flagged for the reviewer (not guessed)

1. **Evaluator interface.** It is unknown whether the held-out test harness expects a
   specific module path / function name / signature. The plan uses a clear, documented
   default (`satad` program + `satad_numpy` wrapper). *If a specific name or signature is
   required, it should be supplied so the port can match it.*
2. **`maxiter` is compile-time** (baked into the unroll). If the evaluator needs it as a
   runtime parameter, that is not expressible in a single field_operator; the port exposes
   10 as the documented fixed value (matching the reference driver).
3. **Agreement tolerance:** targeting `allclose` (~1e-10), not bit-for-bit. Acceptable for
   a physics port; flagged in case exact reproduction is required.

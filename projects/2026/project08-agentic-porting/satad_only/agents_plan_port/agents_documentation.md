# HPC 08 – Agentic Porting

**Progress log: porting ICON's saturation-adjustment kernel to GT4Py**

*Status at end of Step 2 (Fortran read complete) — 18 July 2026*

---

## 1. Instructions I was given

The task is to independently port the standalone saturation-adjustment kernel from
ICON's two-moment microphysics scheme (Fortran) to GT4Py, working in collaborator mode
through a fixed, staged process with approval gates.

### Sources I may use

- **Fortran source:** `satad_only/fortran` — the target kernel to port.
- **GT4Py docs / quickstart:** for learning the DSL.
- **Full-scheme saturation-adjustment file (`fortran/`):** for style/conventions only — a
  different kernel, not the answer.

### Hard constraint — held-out reference

**A finished reference port already exists in the `icon4py/` folder. It is held out for
evaluation and must not be opened, read, or grepped at any point.** If I encounter it by
accident I am to stop and report rather than read it. I have honoured this throughout: I
have not accessed `icon4py/`.

### The agreed process

- **Step 1:** Skim GT4Py `field_operator`/`program` patterns until comfortable with the
  DSL. Stop for approval.
- **Step 2:** Read the target Fortran subroutine(s) closely — inputs, outputs, physical
  meaning, iteration/convergence logic, edge cases, units. Stop for approval.
- **Step 3:** Write a short design plan — file structure, function signatures (units +
  dtypes), handling of the tricky bits (iteration, thresholds, lookup tables). Flag
  genuine uncertainties.
- **Step 4:** Stop and share the plan for review before writing any code.
- **Step 5:** After approval, implement the port; document assumptions and interfaces so an
  independent evaluator can test it without further questions.

**Output must be self-contained:** a separate process will independently test and evaluate
the port afterward, with no access to this conversation.

## 2. What I did

### Getting access to the code

The initial clone attempt failed — the repository was not reachable from my sandbox (it
404'd publicly, and the sandbox has no GitHub credentials). I could not use a personal
access token, as handling credentials/tokens is outside what I am permitted to do. The
working resolution was for you to clone the repo locally on the `jana` branch into the
connected folder, which I then read directly.

### Step 1 — GT4Py DSL familiarisation (completed)

Read the repository's own GT4Py material: `docs/user/next/QuickstartGuide.md` and the
workshop scan-operator exercise. Confirmed the core building blocks needed for this port.

### Step 2 — Reading the target Fortran (completed)

Read the kernel and its dependencies:

- `mo_satad.f90` — the routines `satad_v_3D` (CPU) and `satad_v_3D_gpu` (GPU), plus the
  thermodynamic helper functions.
- `column_driver.f90` — the single-column driver: CSV in → one satad call → CSV out; sets
  `maxiter=10`, `tol=1e-3`.
- `dependencies/*.f90` — `mo_kind` (working precision), `mo_physical_constants` and
  `mo_lookup_tables_constants` (the exact numeric constants).
- `example/fields.csv` — the 10-level sample column (`rho, tk, qv, qc`).

## 3. What I learned

### GT4Py DSL (Step 1)

- **Fields** are defined over named Dimensions (e.g. `Cell`, `K`) and typed as
  `gtx.Field[Dims[...], float64]`.
- **`@field_operator`** = pure, side-effect-free elementwise/reduction code (no in-place
  mutation of arguments).
- **`@program`** = sequences field-operator calls and writes results into `out=` fields.
- **No `if` on fields** — branching is done with `where(mask, true, false)`.
- **No `while` loops** — data-dependent iteration must be a fixed, unrolled count, or a
  `scan_operator` for genuine vertical dependencies.

### The saturation-adjustment kernel (Step 2)

It relaxes each grid point to liquid/vapour equilibrium at constant total density, moving
water between vapour (`qv`) and cloud (`qc`) and adjusting temperature (`T`) by the latent
heat. Key facts:

- **Every level is independent — no vertical coupling.** So the port needs no
  `scan_operator`; it is pure pointwise work.
- **Two branches per point:** (A) if all cloud can evaporate and the air stays
  sub-saturated, set `qv=qw`, `qc=0`, `T=Ttest` directly — no iteration; (B) otherwise a
  Newton iteration on temperature.
- **Newton loop:** `f = twork − T + (Lv/cvd)(qsat(twork) − qv)`;
  `f' = 1 + (Lv/cvd)·dqsat/dT`; stop when `|Δtwork| ≤ tol` or after `maxiter=10`.
- **Closure of branch B:** `T=twork`, `qwa=qsat(T)`, `qc=max(qc+qv−qwa, 1e-20)`, `qv=qwa`.
- **Only the Tetens saturation formula (`ipsat=1`) is active;** the Murphy-Koop branch
  (`ipsat=2`) is dead code needing `tanh`/`log` — not required.
- **Active-path helpers:** `sat_pres_water` (uses `exp`), `qsat_rho`, `dqsatdT_rho`,
  `latent_heat_vaporization`.
- **All arithmetic is `real64` (`float64`).**
- **Interface:** inputs `rho` [kg/m^3]; in-out `te/tk` [K], `qve/qv` [kg/kg], `qce/qc`
  [kg/kg]; controls `maxiter`, `tol` [K].

## 4. What I reported back

- **After Step 1:** the GT4Py building blocks and the key implication that
  iteration/branching need `where` + fixed-count unrolling (or `scan` for vertical
  coupling), pending confirmation from the Fortran.
- **After Step 2:** the full per-point algorithm with units, the two branches, the Newton
  details, and the constants — plus how each Fortran feature maps onto the GT4Py DSL.
  Confirmed no `scan_operator` is needed and that only the Tetens path must be ported.

## 5. Things to watch out for

- **Two different "cp_v" values.** `latent_heat_vaporization` uses a local `PARAMETER
  cp_v = 1850.0`, NOT the physical-constants `cpv = 1869.46`. Using the wrong one silently
  biases every temperature update.
- **Compute `Lv/cvd` (`lwdocvd`) once from the input `T`,** and keep the original `T` and
  `qv` fixed inside the Newton loop (`twork` is a separate working variable). Refreshing
  them mid-loop would change the fixed-point being solved.
- **Port the GPU variant's loop structure, not the CPU `while`-loop.** A fixed `maxiter`
  count with each step masked (converged/non-iterating points freeze) is numerically
  identical to the CPU version and is the only form GT4Py allows.
- **Exact constant transcription.** Several constants are derived (`cvd = cpd−rd = 717.60`;
  `clw = (3.1733+1)·1004.64`; `c5les = 17.269·(273.15−35.86)`). Small transcription errors
  will fail a bit-comparison against Fortran.
- **The 1e-20 floor (`zqwmin`) on `qc`** in branch B must be preserved via a `maximum(...)`.
- **Do not touch `icon4py/`** at any stage — it is the held-out evaluation reference.

## 6. Decisions still to make (for the Step 3 plan)

- **Field layout:** expose the kernel over `(CellDim, KDim)` for generality, or a single
  vertical `KDim` to mirror the single-column driver? Affects the public interface the
  evaluator will call.
- **Newton unrolling mechanism:** rely on GT4Py static `for range()` unrolling for the 10
  steps, or manually unroll / factor a one-step helper `field_operator`? To be confirmed
  against the DSL during implementation.
- **Backend:** embedded/roundtrip for a reference-correct, easily-tested port, versus
  `gtfn_cpu` for performance. Default is likely embedded for verifiability.
- **Convergence semantics to expose:** match the GPU variant exactly (fixed count,
  masked), which we take as the reference behaviour.

---

*Next step: on approval, proceed to Step 3 — the full design plan (file structure,
signatures with units/dtypes, and resolution of the decisions above).*

> **Update:** Step 3 has since been completed; the resulting design plan lives in
> `../agents_plan/satad_gt4py_design_plan.md`. Notably, Decision on Newton unrolling was
> resolved: GT4Py's frontend forbids `for`/`while` loops, so the iteration will be
> unrolled manually (helper step operator called 10 times).

# PR: GT4Py port of ICON saturation adjustment (`satad`)

Independently ports the saturation-adjustment kernel from `satad_only/fortran/mo_satad.f90`
(`satad_v_3D` / `satad_v_3D_gpu`) to the GT4Py `gt4py.next` DSL. The `icon4py/` reference
solution was not opened at any point.

## New files (`satad_only/gt4py/`)

- **`satad_gt4py.py`** — the port: `CellDim×KDim` fields, ICON constants, Tetens helper
  field-operators, the `satad` program, and a `satad_numpy` wrapper.
- **`test_satad_gt4py.py`** — standalone self-check (runs without pytest).
- **`README.md`** — interface, units, assumptions, how to run.
- **`PULL_REQUEST.md`** — this file.

Supporting docs added elsewhere: `satad_only/agents_documentation.md`,
`satad_only/agents_plan.md`, `agents_documentation.docx` (process log + design plan).

## Latest changes

- Added the GT4Py port and the `satad_numpy` numpy wrapper (1-D column or 2-D `(ncells,nlev)`).
- Newton loop manually unrolled `MAXITER=10` times (GT4Py forbids loops); matches the
  `satad_v_3D_gpu` masked structure and the driver's `maxiter`.
- Convergence test written as `(Δt)² > tol²` (the `abs` builtin isn't accepted in that
  comparison) — mathematically identical to `|Δt| > tol`.
- Only the active Tetens path (`ipsat==1`) is ported; Murphy–Koop is dead code, omitted.

## Verification

6/6 self-checks pass: agreement with an independent numpy transcription of the Fortran
`while`-loop (~1e-10), water conservation, condensation-warms/evaporation-cools, both
branches, and 2-D-block == per-column.

## Known limitations

- `maxiter` is compile-time (baked into the unroll), fixed at 10.
- Agreement is ~1e-10, not bit-for-bit (float op ordering in `exp`/division).
- No Fortran compiler in the porting env, so the byte-level `make run` comparison wasn't
  executed here — the numpy transcription stands in for it.

## Repo hygiene

- `.gitignore` extended to cover Python/GT4Py compile artifacts (`__pycache__/`,
  `*.py[cod]`, `*.egg-info/`, `.pytest_cache/`, GT4Py caches, `*.so`, …).
- No changes are made under the vendored `gt4py/` library folder.

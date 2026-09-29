# Running the GT4Py satad port on a performance (compiled) backend

These are notes from an investigation into why passing a performance backend
(e.g. `gtx.gtfn_cpu`) to `satad_numpy(..., backend=...)` had **no effect** — the
kernel kept running on embedded (numpy) execution and GT4Py kept printing
`Using Python execution, consider selecting a performance backend`.

The changes described here were validated end-to-end (all `test_satad_gt4py.py`
tests pass on `gtfn_cpu`) but then **intentionally reverted** so the port stays
a clean, dependency-light reference. This document captures what a future port
needs to do to actually run on a compiled backend.

---

## Symptom

```python
satad_numpy(rho, tk, qv, qc, backend=gtx.gtfn_cpu)
```

still ran embedded and warned about Python execution.

## Root cause #1 — the backend was only used as an *allocator*

The wrapper passed `backend` only to the field buffer allocator:

```python
gtx.as_field([CellDim, KDim], arr.copy(), allocator=backend)
```

The allocator decides **where the array buffers live** — it does *not* select
the execution backend. The `@gtx.program` (`satad`) was still bound to
`backend=None`, so it always ran on embedded (numpy) execution regardless of the
argument.

### Fix

Bind the backend to the *program* before calling it. This is the step that
actually compiles/runs the kernel on that backend:

```python
program = satad.with_backend(backend) if backend is not None else satad
program(te_f, qve_f, qce_f, rho_f, float(tol), offset_provider={})
```

Notes:
- `satad.backend` is `None` by default (embedded). `hasattr(satad, "with_backend")`
  is `True`, and `satad.with_backend(gtx.gtfn_cpu)` returns a backend-bound
  program.
- Alternatively the backend can be set at decoration time:
  `@gtx.program(backend=...)`. `with_backend` is preferable here because it lets
  the wrapper stay backend-agnostic and choose at call time.
- Keep passing `backend` to the allocator *as well* — buffers should live where
  the kernel runs (matters for GPU backends).

## Root cause #2 — bare module-global constants are not resolved by compiled backends

Fixing #1 immediately surfaced a second, hard blocker. The compiled `gtfn`
backend raised at lowering time:

```
gt4py.eve.exceptions.EveValueError:
Symbols {SymbolRef('CP_V'), SymbolRef('C4LES'), SymbolRef('C5LES'),
         SymbolRef('C3LES'), SymbolRef('CLW'), SymbolRef('ZQWMIN'),
         SymbolRef('CVD'), SymbolRef('RV'), SymbolRef('ALV'),
         SymbolRef('TMELT'), SymbolRef('C1ES')} not found.
```

These are the module-level `float` constants (`ALV`, `RV`, `CVD`, …) referenced
*inside* the `@gtx.field_operator` bodies.

**Key insight:** a *compiled* GT4Py backend does not capture bare module-global
Python floats referenced in a field operator. Embedded (numpy) execution happens
to tolerate them as closure variables, which is exactly why the default backend
"worked" and masked the problem — the constants were only ever a problem once a
real backend was selected.

This was confirmed with a minimal reproducer: a one-line field operator
`return a * FOO` with a module-global `FOO = 2.5` compiles fine embedded but
fails with `Symbols {'FOO'} not found` on `gtfn_cpu`.

### What does NOT work

- Bare module globals (`FOO = 2.5`)  → `Symbols not found` on gtfn.
- Plain class attributes (`class Const: FOO = 2.5`; `Const.FOO`)
  → `DSLTypeError: Unexpected object 'Const' of type '<class 'type'>'`.

### What DOES work — a float-subclassing `enum.Enum`

GT4Py only inlines constants that are members of a **`float`-subclassing
`enum.Enum`**. This is the same pattern icon4py uses for all its physical
constants (see
`icon4py/.../muphys/core/common/constants.py`: `class ThermodynamicConsts(ta.wpfloat, enum.Enum)`).

```python
import enum

class C(wpfloat, enum.Enum):   # wpfloat == gtx.float64
    RV = RV
    CVD = CVD
    TMELT = TMELT
    ALV = ALV
    # ...
```

Then reference `C.RV`, `C.CVD`, … inside the field operators instead of the bare
names.

Details:
- The `NAME = NAME` self-reference in the enum body works: the RHS resolves to
  the module global (not yet defined in the class namespace), the LHS becomes the
  enum member. This avoids duplicating the numeric literals — important for the
  *derived* constants (`CVD = CPD - RD`, `CLW = (RCPL+1)*CPD`,
  `C5LES = C3LES*(TMELT-C4LES)`).
- Enum members subclass `float`, so `C.RV + 1.0`, `np.exp(C.RV)`, etc. all still
  behave as plain floats. The original module-level names can be kept untouched
  for pure-numpy code / tests / readability — only the field-operator bodies need
  to switch to `C.<NAME>`.
- Only the constants actually used *inside* field operators need enum members
  (`ALV, CP_V, CLW, TMELT, RV, C1ES, C3LES, C4LES, C5LES, CVD, ZQWMIN`). Base
  constants used only to *compute* others (`RD, CPD, RCPL`) do not.

## Environment gotcha — `BrokenProcessPool` on macOS

Once the code compiled, the first run crashed with:

```
concurrent.futures.process.BrokenProcessPool:
A process in the process pool was terminated abruptly ...
```

This is **not** a code problem. GT4Py's default build uses a `spawn`-based
`ProcessPoolExecutor` (`GT4PY_BUILD_JOBS_MODE=process`). On macOS, `spawn`
re-imports the driver module in the worker, which re-triggers compilation and
kills the worker — the classic `spawn` footgun. Confirmed by switching build
modes:

```bash
export GT4PY_BUILD_JOBS_MODE=THREAD   # ThreadPoolExecutor; keeps parallel builds — recommended
# or
export GT4PY_BUILD_JOBS_MODE=SERIAL   # compile in the calling thread
```

With either of these the C++ compilation succeeds and the kernel runs. `THREAD`
is preferred (keeps build parallelism; avoids re-pickling/re-import).

Related knobs (from `gt4py/src/gt4py/next/config.py`):
- `GT4PY_BUILD_JOBS` — number of parallel compile jobs (`<=0` forces serial).
- `GT4PY_BUILD_CACHE_DIR` — where compiled stencils are cached.
- `PYTHONOPTIMIZE=1` (or `python -O`) — silences a separate "not running in
  optimized mode" performance warning.

Toolchain present on the dev machine when this worked: `/usr/bin/gcc`,
`/usr/bin/g++`, cmake from the venv, and the bundled `gridtools_cpp` headers
(no system Boost needed for `gtfn_cpu`).

---

## Checklist for a future port that must run compiled

1. In the numpy wrapper, bind the backend to the program:
   `prog = kernel.with_backend(backend) if backend else kernel`, and keep passing
   `backend` as the `as_field(..., allocator=backend)`.
2. Move every constant referenced *inside* a `field_operator` into a
   `class C(wpfloat, enum.Enum)` and reference it as `C.<NAME>`. Do not rely on
   bare module globals or plain class attributes.
3. Run with `GT4PY_BUILD_JOBS_MODE=THREAD` (at least on macOS) to avoid the
   `BrokenProcessPool` crash.
4. Verify: the `Using Python execution` warning should disappear for the
   backed calls, and results should match the embedded/numpy reference to the
   same tolerances.

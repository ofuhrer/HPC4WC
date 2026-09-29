# ICON4Py Agent Instructions

ICON4Py is a Python implementation of the Fortran [ICON climate and weather model](https://www.icon-model.org/). The upstream open source release is available at https://gitlab.dkrz.de/icon/icon-model.

## Coding guidelines

Always read the CODING_GUIDELINES.md document first and follow it.

## Monorepo structure

uv workspace with 11 namespace packages. All share the `icon4py` namespace. Source lives under `<package>/src/icon4py/...`. Packages are installed editable by `uv sync`.

```
model/
  atmosphere/
    advection/          # icon4py.model.atmosphere.advection
    diffusion/          # icon4py.model.atmosphere.diffusion
    dycore/             # icon4py.model.atmosphere.dycore
    subgrid_scale_physics/
      microphysics/     # icon4py.model.atmosphere.subgrid_scale_physics.microphysics
      muphys/           # icon4py.model.atmosphere.subgrid_scale_physics.muphys
  common/               # icon4py.model.common  ← shared code, all model packages depend on this
  driver/               # icon4py.model.driver  ← depends on diffusion, dycore, common, testing
  standalone_driver/    # icon4py.model.standalone_driver
  testing/              # icon4py.model.testing ← pytest plugin, fixtures, serialbox helpers
tools/                  # icon4py.tools ← Fortran integration (py2fgen CLI), independent of model
bindings/               # icon4py.bindings ← Fortran wrappers for diffusion/dycore/muphys, depends on tools.py2fgen
```

Tach enforces the dependency graph in `tach.toml`. All model atmosphere packages and standalone_driver depend only on `common`. Driver depends on diffusion + dycore + common + testing. Tools is independent. Bindings depends on diffusion + dycore + muphys + common + tools.py2fgen.

**Always run `uv sync` from the repo root.** Running it from a subpackage only installs that package's deps.

## Key architecture

icon4py implements the ICON atmospheric model using GT4Py stencil operators (`field_operator`, `scan_operator`).

### Grid topology

- ICON grids are built from a base shape (icosahedron or torus). Icosahedral grids are refined by recursive bisection; a grid with root R and bisection level B has R^2 * 4^B cells per original face.
- Fields live on three topological entities: **cells** (triangle centers), **edges** (triangle sides), and **vertices** (triangle corners). Neighbor connectivities encode the mesh structure: C2E (cell-to-edge), E2C (edge-to-cell), V2E (vertex-to-edge), C2E2C, E2C2V, E2C2E, etc.

### Distributed memory

- Each MPI rank owns a subset of cells/edges/vertices plus halo regions (ghost cells from neighboring ranks). Owner masks distinguish owned from halo entries.
- After stencil computations that write to halo-adjacent fields, halo values must be exchanged via GHEX before neighbor accesses are valid.
- `GlobalReductions` wraps `MPI.Allreduce` for min/max/sum/mean.
- `h_grid.Zone` defines domain regions ordered from most inclusive (`END` = all local including halos) to most restrictive (`INTERIOR`). Key zones: `END` > `HALO` > `LOCAL` > `INTERIOR`.

## Environment setup

Python is managed by uv (see `.python-version` for the required version). Install system dependencies first, then sync the workspace:

```bash
uv sync --extra all
```

### System dependencies

- C/C++/Fortran compilers (gcc/g++, gfortran)
- CMake, pkg-config
- git

The `distributed` extra (mpi4py, ghex) additionally requires Boost (headers) and an MPI implementation (e.g. OpenMPI).

If a `shell.nix` exists in the repo root, you can use it to provide these dependencies via nix.

### Relevant environment variables

- `ICON4PY_TEST_DATA_PATH`: where serialized test data is stored
- `GT4PY_BUILD_CACHE_DIR`: GT4Py stencil compilation cache location
- `GT4PY_BUILD_JOBS`: limit parallel stencil compilation jobs (unset by default)
- `PYTEST_ADDOPTS`: default pytest options (xdist workers, verbosity)

### Clean rebuild

```bash
rm -rf .venv && uv sync --extra all --no-cache
```

uv caches built wheels by package hash, not just Python version. If native extensions fail to load (e.g. `ModuleNotFoundError` for a `.so`), the cache may contain a build for a different Python version. Fix: re-run the clean rebuild command above.

## Pre-commit checks

All checks (ruff lint, ruff format, tach, mypy, license headers, toml/yaml formatting) run through pre-commit:

```bash
uv run --group dev --frozen --isolated pre-commit run --all-files
```

Notes:

- License headers are auto-added by the `insert-license` hook using `HEADER.txt`. Do not add them manually.
- mypy uses `--group typing-distributed` (includes mpi4py stubs), not just `--group typing`.
- mypy only runs on specific paths listed in `pyproject.toml [tool.mypy].files`. Running mypy on the whole codebase will produce errors from unchecked packages.
- ruff excludes `docs/` and `examples/` directories.

## Tests

### Test categories

| Category      | Marker / flag               | Needs serialized data?    | Typical runtime        |
| ------------- | --------------------------- | ------------------------- | ---------------------- |
| Unit tests    | none (no `datatest` marker) | No                        | Fast (\<1s per test)   |
| Data tests    | `@pytest.mark.datatest`     | Yes (auto-downloaded)     | Slow (2+ min per file) |
| Stencil tests | `-k stencil_tests`          | No (uses generated grids) | Moderate               |
| MPI tests     | `@pytest.mark.mpi`          | Yes (auto-downloaded)     | Slow                   |

### Running tests

```bash
# Fast unit tests (no data required):
uv run --group test --frozen pytest model/<component>/tests/<component>/unit_tests/

# Skip datatests:
uv run --group test --frozen pytest --datatest-skip model/<component>/

# Only datatests (requires test data):
uv run --group test --frozen pytest --datatest-only model/<component>/

# MPI tests (requires mpi4py, distributed extra; always use -n0 for sequential):
mpirun -np 4 ci/scripts/ci-mpi-wrapper.sh uv run --group test --frozen pytest -v -s --with-mpi -n0 -k mpi_tests model/<component>/
#   --with-mpi: enables MPI test mode (from pytest-mpi plugin)
#   -k mpi_tests: selects tests by directory/class name convention
#   ci-mpi-wrapper.sh: suppresses stdout from non-zero ranks

# Sequential mode (reduce memory pressure):
uv run --group test --frozen pytest -n0 <paths>
```

### Custom pytest options

Registered by `icon4py.model.testing.pytest_hooks` (auto-loaded via `addopts`):

| Option                                      | Description                                                                       |
| ------------------------------------------- | --------------------------------------------------------------------------------- |
| `--datatest-only`                           | Run only `@pytest.mark.datatest` tests                                            |
| `--datatest-skip`                           | Skip all datatests                                                                |
| `--backend <name>`                          | GT4Py backend (default: roundtrip; others: gtfn_cpu, gtfn_gpu, embedded)          |
| `--grid <name>`                             | Grid to use                                                                       |
| `--enable-mixed-precision`                  | Switch from double to mixed-precision                                             |
| `--level {any,unit,integration,validation}` | Filter by `@pytest.mark.level` marker. `any` (default) excludes validation tests. |
| `--skip-stenciltest-verification`           | Skip verification of StencilTest against reference outputs                        |

### Test directory convention

Each component's tests live under `tests/<component_name>/` with subfolders:

```
tests/<component>/
  unit_tests/         # fast, no external data
  stencil_tests/      # GT4Py stencil integration tests (for icon-exclaim)
  integration_tests/  # end-to-end tests
  mpi_tests/          # parallel/MPI tests
  conftest.py
  fixtures.py
```

### Benchmarking

`pytest-benchmark` is active by default. Use `--benchmark-disable` to skip benchmarks during regular test runs.

### Fixture imports

Fixtures re-exported through a component's `fixtures.py` must be explicitly imported in the test file, even if not used directly in the test body. Pytest resolves fixtures from the test module's namespace, not transitively through other fixtures. Missing imports cause `fixture 'X' not found` errors.

## GT4Py stencil compilation

GT4Py compiles stencils to C++ on first use. This is slow. To limit parallel compilation jobs:

```bash
export GT4PY_BUILD_JOBS=4
```

## Nox (CI-style test sessions)

Nox mirrors the CI pipeline. Useful for running comprehensive test suites:

```bash
# List sessions:
uv run --group test --frozen nox -l

# Run all tests for a specific component and subset:
uv run --group test --frozen nox -s 'test_common(datatest=True)'
uv run --group test --frozen nox -s 'test_common(datatest=False)'
```

Subset options: `datatest`, `stencils`, `basic` (datatest-skip, no stencils/benchmarks).

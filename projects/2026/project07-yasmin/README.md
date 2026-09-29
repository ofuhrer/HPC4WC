# YASMIN

**Yet Another Stencil Mini Intermediate Notation**

Yasmin is a small experimental Python DSL and compiler for stencil computations.
A stencil is expressed once and can be executed with NumPy or lowered to native
C++ and OpenMP code.

The project is part of the
[High Performance Computing for Weather and Climate](https://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=199803&semkez=2026S&lang=en&ansicht=ALLE)
course at ETH Zurich.

> [!NOTE]
> Yasmin is an experimental course project. Its API and compiler internals are
> still evolving.

## Installation

Yasmin requires Python 3.11+ and NumPy 2.0+.

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

The C++ and OpenMP backends additionally require a compatible native compiler.

## Example

```python
import numpy as np
import yasmin as yasi
from yasmin.frontend.expr import SymbolicExpr

x, y = yasi.Dimension("x", "y")
u = yasi.Field("u", dims=(x, y), dtype=yasi.float64)
out = yasi.Field("out", dims=(x, y), dtype=yasi.float64)


@yasi.stencil
def laplace(f: yasi.Field) -> SymbolicExpr:
    return (
        f[-1, 0] + f[1, 0] + f[0, -1] + f[0, 1] - 4.0 * f[0, 0]
    )


@yasi.operator
def apply(out: yasi.Field, u: yasi.Field) -> None:
    out[0, 0] = laplace(u)


u_data = np.arange(64, dtype=np.float64).reshape(8, 8)
out_data = np.zeros_like(u_data)

yasi.execute(
    apply(out, u),
    backend="numpy",
    fields={u: u_data, out: out_data},
)
```

Field indices are relative offsets from the current grid point. Yasmin infers
the required halo and executes only where all stencil accesses are valid.

Available execution backends are `numpy`, `cpp`, and `openmp`.

## More

- [`HPC4WC_Yasmin.pdf`](HPC4WC_Yasmin.pdf) — final project report
- [`examples/laplacian.py`](examples/laplacian.py) — small runnable example
- [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb) — guided introduction
- [`docs/design.md`](docs/design.md) — design notes and compiler architecture
- [`benchmarks/`](benchmarks/) — benchmark implementations and results

Run the test suite with:

```bash
pytest
```

# Yasmin Design Notes

The core idea is to separate stencil semantics from execution.

```text
Python DSL
    |
    v
Stencil IR
    |
    +-----------> NumPy
    |
    v
Loop IR
    |
    +-----------> C++
    |
    +-----------> OpenMP
    |
    +-----------> CuPy        # later
    |
    +-----------> CUDA        # later
```

## Core abstractions

The user-facing stencil abstractions are:

- `Dimension`
- `Field`
- `Scalar`
- `SymbolicExpr`
- `Stencil`
- `Operator`

A stencil describes a reusable local computation.

An operator describes one or more updates over a domain.

Example syntax:

```python
import yasmin as yasi

x, y = yasi.Dimension("x", "y")
u = yasi.Field("u", dims=(x, y), dtype=yasi.float64)
out = yasi.Field("out", dims=(x, y), dtype=yasi.float64)
alpha = yasi.Scalar("alpha", dtype=yasi.float64)


@yasi.stencil
def laplace(u: yasi.Field) -> yasi.SymbolicExpr:
    return (
        u[-1, 0]
        + u[1, 0]
        + u[0, -1]
        + u[0, 1]
        - 4.0 * u[0, 0]
    )


@yasi.operator
def heat(out: yasi.Field, u: yasi.Field, alpha: yasi.Scalar) -> None:
    out[0, 0] = u[0, 0] + alpha * laplace(u)
```

## Stencil IR

The Stencil IR is the main backend-independent semantic representation.

Field accesses are represented relative to the current logical iteration point.

For example:

```python
u[-1, 0]
```

is normalized to something conceptually equivalent to:

```text
FieldAccess(
    field=u,
    offsets=(-1, 0),
)
```

The IR contains only a small set of nodes:

```text
Expr
├── Literal
├── ScalarRef
├── FieldAccess
└── BinaryExpr

Stmt
└── Assign
```

Fields, dimensions, and scalars are declaration objects rather than expression nodes.

## Analysis

Stencil accesses contain enough information to derive properties of the computation.

For example:

```text
u@(-2, 0)
u@(+1, 0)
u@(0, +3)
```

implies:

```text
x offsets: [-2, +1]
y offsets: [ 0, +3]
```

This can be used to infer:

- valid iteration domains;
- halo requirements;
- read/write sets;
- later, distributed halo exchanges.

## Loop IR

The native backends lower Stencil IR into a lower-level Loop IR.

The Loop IR represents concepts such as:

- loops;
- indices;
- loads;
- stores;
- explicit memory accesses;

Conceptually:

```text
Stencil IR

out@(0, 0) = u@(-1, 0) + u@(+1, 0)

        |
        v

Loop IR

for i = 1 .. nx - 1:
    out[i] = u[i - 1] + u[i + 1]
```

NumPy executes directly from Stencil IR using array slices and does not pass through Loop IR. OpenMP parallelization is added during native code generation.

## Backends

The backend progression is:

1. NumPy
2. C++
3. OpenMP
4. CuPy *(planned)*
5. CUDA *(planned)*

The same Stencil IR should drive all backends.

## Future extensions

Possible later extensions include:

- symbolic discretization of PDEs;
- time-dependent fields;
- boundary conditions;
- operator fusion and scheduling;
- MPI and `mpi4py`;
- CUDA-aware MPI;
- MLIR-based lowering.

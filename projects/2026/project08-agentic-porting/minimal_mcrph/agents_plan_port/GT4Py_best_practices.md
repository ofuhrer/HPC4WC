# GT4Py Best Practices & Common Pitfalls

## Field Operators (FO) - Core Rules

| Rule | Why | Example |
|------|-----|---------|
| **Pure functions** | FOs cannot have side effects | Never modify arguments, only return new values |
| **No Python loops over fields** | GT4Py needs to analyze operations statically | Use vectorized operations or `@gtx.scan_operator` |
| **No `if` statements with field conditions** | Use `where(condition, x, y)` instead | ❌ `if cond: x else: y` → ✅ `where(cond, x, y)` |
| **No mutation of arguments** | FO arguments are inputs only | ❌ `a[i] = b[i]` → ✅ `return b` |

## Programs - The Execution Layer

| Rule | Why | Example |
|------|-----|---------|
| **Programs can mutate** | Unlike FOs, programs can modify fields in-place | Use `out=` parameter to specify output fields |
| **Explicit domains required** | GT4Py needs to know which grid cells to process | `domain={dims.CellDim: (start, end), dims.KDim: (kstart, kend)}` |
| **No return value** | Programs modify fields via `out` parameter | Programs are called for side effects |

## Common Syntax Patterns (Fortran → GT4Py)

| Fortran Pattern | GT4Py Equivalent |
|-----------------|------------------|
| `result = a + b` | `result = a + b` (same!) |
| `if (cond) x = y else z = w` | `x = where(cond, y, z)` |
| `a(i) = b(i) + c` | `a = b + broadcast(c, (dim,))` |
| Nested loops over K | Use `@gtx.scan_operator` for vertical sweeps |
| In-place modification | Use program with `out=` field |

## Critical "Don'ts" for GT4Py

```python
# ❌ DON'T: Use Python loops over fields
for k in range(n):
    temp[k] = a[k] + b[k]  # Will NOT work correctly

# ✅ DO: Use vectorized operations
temp = a + b

# ❌ DON'T: Mutate arguments in field operators
def bad_fo(a, b):
    a = a + b  # This rebinds, but pattern is confusing
    return a

# ❌ DON'T: Use Python conditionals with fields
def bad_cond(a, b, cond):
    if cond > 0:  # This won't work as expected
        return a
    else:
        return b

# ✅ DO: Use where for conditional logic
def good_cond(a, b, cond):
    return where(cond > 0, a, b)
```

## Scan Operators (Vertical Sweeps)

```python
@gtx.scan_operator(axis=dims.KDim, forward=True, init=(0.0, 0.0))
def vertical_sweep(state_kup, current_level_data):
    # state_kup: tuple of values from previous level
    # current_level_data: values at current level
    # Return new state for next level
    return (new_value1, new_value2)
```

**Use cases**: Sedimentation, implicit time integration, vertical physics

---

## Type System in GT4Py & icon4py

### Type Aliases

icon4py provides a well-structured type alias system in `icon4py.model.common.type_alias`:

| Type Alias | Description |
|-----------|-------------|
| `wpfloat` | Working precision float (default `gtx.float64`) |
| `vpfloat` | Variable precision float (can be float32 or float64) |
| `anyfloat` | Union of float32 and float64 |

**Precision configuration**:
```python
# Available precisions: "double", "mixed", "single"
ta.set_precision("double")  # default
```

### Field Type Aliases

Fields are defined with dimensions and type parameters in `icon4py.model.common.field_type_aliases`:

```python
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common import type_alias as ta
from icon4py.model.common import dimension as dims

# 1D fields
CellField: Field[Dims[CellDim], T]
EdgeField: Field[Dims[EdgeDim], T]
VertexField: Field[Dims[VertexDim], T]
KField: Field[Dims[KDim], T]

# 2D fields (horizontal + vertical)
CellKField: Field[Dims[CellDim, KDim], T]  # Most common for column physics
EdgeKField: Field[Dims[EdgeDim, KDim], T]
VertexKField: Field[Dims[VertexDim, KDim], T]
```

### Type Variable for Generic Fields

```python
from typing import TypeVar
T = TypeVar("T", wpfloat, vpfloat, float, bool, gtx.int32, gtx.int64)

# Use T for generic type hints that work with any supported type
def generic_process(data: fa.CellKField[T]) -> fa.CellKField[T]:
    return data * 2
```

### Type Hints & Annotations

```python
# ✅ DO: Use proper field type aliases
def process(
    temperature: fa.CellKField[ta.wpfloat],
    density: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    return temperature * density

# For scalar parameters in FOs, use the type directly
@gtx.field_operator
def scalar_op(temp: ta.wpfloat, factor: ta.wpfloat) -> ta.wpfloat:
    return temp * factor
```

## Data Types in GT4Py

| Type | GT4Py Usage | icon4py Alias |
|------|-------------|---------------|
| `wpfloat` | Use `ta.wpfloat` (working precision float) | `type_alias.wpfloat` |
| `bool` | Use `bool` for masks | `bool` |
| `int32` | Use `gtx.int32` for enum values, loop counters | `gtx.int32` |
| `int64` | Use `gtx.int64` for large integers | `gtx.int64` |
| Scalars | Pass directly, no field wrapper needed | Any numeric type |

---

## Constants in icon4py

### Two-Tier Constants System

icon4py uses a two-tier approach for constants:

| Tier | Purpose | Location | Example |
|------|---------|----------|---------|
| **Module-level `Final[ta.wpfloat]`** | Module-wide constants | `icon4py.model.common.constants` | `GAS_CONSTANT_DRY_AIR: Final[ta.wpfloat] = 287.04` |
| **`enum.Enum` inheriting from `ta.wpfloat`** | GT4Py FO constants | `microphysics_constants.py` | `class PhysicsConstants(ta.wpfloat, enum.Enum)` |

### Module-Level Constants

**File**: `icon4py/model/common/constants.py`

```python
from typing import Final
from icon4py.model.common import type_alias as ta

# Physical constants
GAS_CONSTANT_DRY_AIR: Final[ta.wpfloat] = 287.04
SPECIFIC_HEAT_CAPACITY_PRESSURE_DRY_AIR: Final[ta.wpfloat] = 1004.64
MELTING_TEMPERATURE: Final[ta.wpfloat] = 273.15
LATENT_HEAT_FOR_VAPORISATION: Final[ta.wpfloat] = 2.5008e6
GRAVITATIONAL_ACCELERATION: Final[ta.wpfloat] = 9.80665

# Common aliases
RD = GAS_CONSTANT_DRY_AIR
CPD = SPECIFIC_HEAT_CAPACITY_PRESSURE_DRY_AIR
tmelt = MELTING_TEMPERATURE
```

### FO Constants via Enum

**File**: `icon4py/model/atmosphere/subgrid_scale_physics/microphysics/microphysics_constants.py`

```python
import enum
from icon4py.model.common import type_alias as ta
from icon4py.model.common.constants import PhysicsConstants

class MicrophysicsConstants(ta.wpfloat, enum.Enum):
    """
    Constants used for microphysics computations.
    Inherits from ta.wpfloat to work as scalars in field operators.
    """
    TETENS_P0 = 610.78
    TETENS_AW = 17.269
    TETENS_BW = 35.86
    TETENS_DER = TETENS_AW * (PhysicsConstants.tmelt - TETENS_BW)
    TETENS_AI = 21.875
    TETENS_BI = 7.66
    THRESHOLD_FREEZE_TEMPERATURE = 271.15
    QMIN = 1.0e-15
    # ... more constants

    # Can reference PhysicsConstants too
    RCPD = 1.0 / PhysicsConstants.cpd
```

### Usage in Field Operators

```python
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics.microphysics_constants import MicrophysicsConstants

@gtx.field_operator
def sat_pres_water(temperature: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    """Saturation vapor pressure over water (Tetens formula)."""
    return MicrophysicsConstants.TETENS_P0 * exp(
        MicrophysicsConstants.TETENS_AW
        * (temperature - PhysicsConstants.tmelt)
        / (temperature - MicrophysicsConstants.TETENS_BW)
    )
```

### Configuration Enums

**File**: `icon4py/model/atmosphere/subgrid_scale_physics/microphysics/microphysics_options.py`

```python
import enum
import gt4py.next as gtx

class LiquidAutoConversionType(gtx.int32, enum.Enum):
    """Options for computing liquid auto conversion rate"""
    KESSLER = 0
    SEIFERT_BEHENG = 1

class SnowInterceptParameterization(gtx.int32, enum.Enum):
    """Options for deriving snow intercept parameter"""
    FIELD_BEST_FIT_ESTIMATION = 1
    FIELD_GENERAL_MOMENT_ESTIMATION = 2
```

**Usage**:
```python
def get_autoconversion_rate(
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    autoconv_type: LiquidAutoConversionType,  # Passed as scalar
) -> fa.CellKField[ta.wpfloat]:
    if autoconv_type == LiquidAutoConversionType.KESSLER:
        return compute_kessler(qc)
    else:
        return compute_seifert_beheng(qc, qv)
```

---

## Constants as Arguments

For constants used in FOs that need to be configurable:

```python
# For constants used in FOs, pass as regular arguments
@gtx.field_operator
def compute_something(constant_param: ta.wpfloat, temp: fa.CellKField[ta.wpfloat]):
    return temp * constant_param

# In program setup:
model_options.setup_program(
    backend=backend,
    program=my_program,
    constant_args={"constant_param": 1.23},  # Pass constants here
)
```

---

## Dataclasses for Structured Data

**Use for configs, metric states, and data containers:**

```python
import dataclasses
from icon4py.model.common import type_alias as ta

@dataclasses.dataclass(frozen=True)
class SaturationAdjustmentConfig:
    """Configuration for saturation adjustment"""
    max_iter: int = 10
    tolerance: ta.wpfloat = 1.0e-3

@dataclasses.dataclass
class MetricStateSaturationAdjustment:
    """Metric state needed for saturation adjustment"""
    ddqz_z_full: fa.CellKField[ta.wpfloat]
```

**Usage in class**:
```python
class SaturationAdjustment:
    def __init__(
        self,
        config: SaturationAdjustmentConfig,
        grid: icon_grid.IconGrid,
        vertical_params: v_grid.VerticalGrid,
        metric_state: MetricStateSaturationAdjustment,
        backend: gtx_typing.Backend | None,
    ):
        self.config = config
        # ... initialize other components
```

---

## Error Handling

```python
# ❌ DON'T: Use Python print() in FOs
def bad_fo(a):
    print("debug")  # Won't work as expected
    return a

# ✅ DO: Use assert for debugging (only in development)
def good_fo(a):
    assert a >= 0, "Negative values not allowed"
    return a
```

---

## Testing & Verification

| Strategy | How |
|----------|-----|
| Compare to Fortran | Run same test cases, compare output fields |
| Unit tests | Test individual FOs with small inputs |
| Integration tests | Test full driver behavior |
| Convergence tests | For iterative schemes (like satad), verify convergence |

---

## Quick Reference: Most Common Patterns

### 1. Element-wise operation
```python
@gtx.field_operator
def elementwise_op(a: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    return a * 2.0 + 1.0
```

### 2. Multi-field operation
```python
@gtx.field_operator
def multi_field_op(a, b, c) -> fa.CellKField[ta.wpfloat]:
    return (a + b) * c
```

### 3. Conditionals
```python
@gtx.field_operator
def conditional(a, b, cond) -> fa.CellKField[ta.wpfloat]:
    return where(cond > 0, a, b)
```

### 4. Reductions
```python
@gtx.field_operator
def vertical_sum(a: fa.CellKField[ta.wpfloat]) -> ta.wpfloat:
    return gtx.sum(a, axis=dims.KDim)
```

### 5. Program with domain
```python
@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def my_program(
    a: fa.CellKField[ta.wpfloat],
    b: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    elementwise_op(a, out=b)
```

---

## Import Summary

```python
# Core imports for microphysics ports
import gt4py.next as gtx
from icon4py.model.common import type_alias as ta
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common import dimension as dims
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics.microphysics_constants import MicrophysicsConstants
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.common import model_options

# For minimal 2-moment scheme
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.grid import vertical as v_grid
from typing import Final, TypeVar
from dataclasses import dataclass
```

---

## GT4Py File Structure Best Practices

### Key Insight: Group Related Stencils

When porting the ICON microphysics to GT4Py, follow icon4py's pattern of grouping related field operators into a small number of files:

**DO** (icon4py approach - recommended):
```
microphysics/
├── stencils/
│   ├── __init__.py
│   ├── microphysical_processes.py   # All process operators
│   └── graupel_stencils.py          # Graupel-specific operators
```

**DON'T** (over-partitioning):
```
microphysics/
├── stencils/
│   ├── ccn.py
│   ├── nucleation.py
│   ├── freezing.py
│   ├── deposition.py
│   ├── melting.py
│   └── ... (many small files)
```

### Why This Works Better

1. **icon4py precedent**: The existing `single_moment_six_class_gscp_graupel` uses one `graupel_stencils.py` and one `microphysical_processes.py` for shared operators.

2. **Our scheme is simpler**: The minimal 2-moment scheme has only ~7 core processes, all relatively small.

3. **Easier maintenance**: One file to find all process operators instead of scattered files.

4. **Cleaner imports**: Import from a single `microphysical_processes` module.

### File Structure for Minimal 2-Moment Port

```
minimal_mcrph/
├── stencils/
│   ├── __init__.py
│   ├── driver_stencils.py           # Driver programs (minimal)
│   └── microphysical_processes.py   # All process field operators
```

### Naming Conventions

| File | Purpose | Example Contents |
|------|---------|------------------|
| `driver_stencils.py` | Programs that orchestrate the driver | `minimal_mcrph_run` |
| `microphysical_processes.py` | All process field operators | `ccn_activation`, `ice_nucleation`, `cloud_freeze`, etc. |
| `saturation_adjustment_stencils.py` | Satad operators | `compute_subsaturated_case`, `update_temperature`, etc. |

### Import Organization

```python
# In stencils/__init__.py
from .driver_stencils import minimal_mcrph_run
from .microphysical_processes import (
    ccn_activation_hdcp2,
    ice_nucleation_homhet,
    cloud_freeze,
    vapor_dep_relaxation,
    ice_melting,
    set_default_n,
)

__all__ = [
    "minimal_mcrph_run",
    "ccn_activation_hdcp2",
    "ice_nucleation_homhet",
    "cloud_freeze",
    "vapor_dep_relaxation",
    "ice_melting",
    "set_default_n",
]
```

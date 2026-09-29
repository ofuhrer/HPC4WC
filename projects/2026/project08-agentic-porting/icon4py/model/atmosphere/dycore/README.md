# icon4py-atmosphere-dycore

## Description

Contains code ported from ICON `src/atm_dyn_iconam`, which is the dynamical core of the ICON model.

## Treatment of stencils operating on different vertical levels

Variables in ICON can either sit on full levels or half levels. Let's denote the number of full levels is nlev.
Full-level variables have nlev levels in vertical dimension, while half-level variables have nlev+1 in vertical dimension.
When running a GT4Py program over nlev+1 levels, do not output full-level and half-level variables together in a field_operator until the upper vertical bound of nlev+1 because the nlev+1 level will also be written for a full-level variable, which is out-of-bounds.
The following code is an example:

```python
@field_operator
def _foo(
    input_var: CellKField[wpfloat],
    k_field: KField[wpfloat],
    nlev: int32,
) -> tuple[CellKField[wpfloat], CellKField[wpfloat]]:
    (half_level_var, full_level_var) =  where(
        (k_field >= 0) & (k_field < nlev),
        _compute_var(input_var),
        (half_level_var, full_level_var),
    )
    half_level_var = where(k_field == nlev, _init_cell_kdim_field_with_zero_vp(), half_level_var)
    return half_level_var, full_level_var

@program
def foo(
    input_var: CellKField[wpfloat],
    k_field: KField[wpfloat],
    nlev: int32,
    full_level_var: CellField[wpfloat],
    half_level_var: CellField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
) -> None:
    _foo(
        input_var,
        k_field,
        nlev,
        out=(half_level_var, full_level_var),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        }
    )
)

foo(
    input_var=input_var,
    full_level_var=full_level_var,
    half_level_var=half_level_var,
    k_field=k_field,
    nlev=nlev,
    horizontal_start=start_cell,
    horizontal_end=end_cell,
    vertical_start=0,
    vertical_end=nlev + 1,
    offset_provider={},
)
```

Here is a fix to the example above:

```python
@program
def foo(
    input_var: CellKField[wpfloat],
    full_level_var: CellField[wpfloat],
    half_level_var: CellField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
) -> None:
    _compute_var(
        input_var,
        out=(half_level_var, full_level_var),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end - 1),
        }
    )
    _init_cell_kdim_field_with_zero_vp(
        out=half_level_var,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_end - 1, vertical_end),
        }
    )
)

foo(
    input_var=input_var,
    full_level_var=full_level_var,
    half_level_var=half_level_var,
    horizontal_start=start_cell,
    horizontal_end=end_cell,
    vertical_start=0,
    vertical_end=nlev + 1,
    offset_provider={},
)
```

In the fix above, the original field_operator `_foo` is removed and the GT4Py program directly calls `_compute_var` and `_init_cell_kdim_field_with_zero_vp` separately with different vertical bounds.
The illegal access to nlev+1 level of the full-level variable can thus be avoided by calling `_compute_var` with vertical bounds of (0, nlev).
The computation on nlev+1 level `half_level_var = where(k_field == nlev, _init_cell_kdim_field_with_zero_vp(), half_level_var)` in the removed field_operator `_foo` is instead performed explicitly by calling `_init_cell_kdim_field_with_zero_vp` with another vertical bounds of (nlev, nlev+1).

## Installation instructions

Check the `README.md` at the root of the `model` folder for installation instructions.

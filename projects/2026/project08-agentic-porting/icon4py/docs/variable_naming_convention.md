# Naming convection of variables in ICON4Py

This naming convention should help to have a common style for variable names to improve the understandability of variables.

## Structure of name

In a discretized domain, it is important to know where a variable is located.
ICON uses unstructured grid with the computational domain splitted into a finite number of triangular prisms. In the horizontal coordinates, all quantities in ICON sit either at cell centers, edges, or vertices.
Currently, it is recommendded that we add a suffix, `_at_cells`, `_at_edges`, or `_at_vertices`, to a variable to let users understand the location at which the variable is defined.

The vertical coordinate is divided into a discrete number of levels and a variable can either sit on the so-called full levels (model levels) or k-half levels (interface of two stacking neighboring triangular prisms). We further add another suffix `_on_model_levels` or `_on_half_levels`to show that a variable is defined on model or k-half levels.

We do not add any suffix to all fundamental variables, for instance the prognostic variables updated by the model via solving some the Navier-Stokes equations, to indicate their location. The diagnostic quantities derived from those fundamental variables, for instance via interpolation, should be added with the suffixes as stated above to indicate where they are defined.

However, when there is no ambiguity, the suffix indicating its horizontal or vertical location can be omitted for a derived variable . For instance, we do not need to add `_at_cells`, `_at_edges`, or `_at_vertices` to `vn_on_half_levels` in icon4py because `vn` is only defined at edges.

## Example

The vertical wind speed on k-half levels with the contravariant correction due to the terrain-following coordinates is

```
contravariant_corrected_w_at_cells_on_half_levels
```

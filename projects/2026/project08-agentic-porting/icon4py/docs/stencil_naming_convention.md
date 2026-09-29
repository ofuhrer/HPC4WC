# Naming convection of stencils in ICON4Py

This naming convention should help to have a common style for stencil names.
It should help to read and understand stencil code.

## Multiple-word identifier format

The stencils must follow snake case.

## Length of name

The program name of the stencil needs to have less then 70 characters.

## Structure of name

The `field_operator` name should start with an underscore character.
The `program` name should be the same as for the field_operator without the underscore.
The `program` name should start with a verb.
Commonly used verbs are accumulate, add, apply, compute, copy, correct, extrapolate, interpolate, return, set, and
solve.
The verb can be followed by more describing words.
If possible one should not use the names of variables that are part of the signature already as describing words.

## Example

A good example for a stencil name is for the `field_operator name`:

```
_interpolate_to_cell_center
```

and for the program

```
interpolate_to_cell_center
```

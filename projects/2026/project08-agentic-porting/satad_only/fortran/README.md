# satad_only / fortran

A single-column sandbox for ICON's **saturation adjustment** (`mo_satad.f90`)
on its own -- the cut-down sibling of the full 2-moment microphysics sandbox
in [`../../fortran`](../../fortran). It exists to make the GT4Py port
tractable: satad is a small, self-contained piece of the scheme, so porting it
first is a good way to get a feel for the workflow before tackling the whole
thing.

Saturation adjustment relaxes each level to thermodynamic equilibrium at
constant total density -- moving water between vapour (`qv`) and cloud water
(`qc`) and adjusting temperature (`tk`) by the associated latent heat, via a
per-level Newton iteration. It treats every level independently, so there is
no vertical coupling: no half-level heights, layer thickness or pressure are
needed, and (unlike the full scheme) there is no lookup-table file I/O and
hence **no netCDF dependency**.

## Layout

```
column_driver.f90            top-level program: CSV in -> satad call -> CSV out
mo_satad.f90                 the saturation-adjustment routine itself
dependencies/*.f90           the infrastructure / physical-constant modules
                             satad needs to compile (mo_kind,
                             mo_physical_constants, mo_lookup_tables_constants)
example/                     sample input column (fields.csv) and where the
                             driver writes output_fields.csv
Makefile                     build rules (see below)
```

## Dependencies

Just a Fortran compiler -- no netCDF. On macOS with
[Homebrew](https://brew.sh): `brew install gcc` (provides `gfortran`). On
Linux, install your distribution's `gfortran` package.

## Building

The default `FC` in the Makefile is `gfortran`. If it isn't on your `PATH`,
pass `FC=` explicitly, e.g. `make check FC=gfortran-16`.

```
make check     # fast syntax/interface check only, no code generation
make           # full compile to build/obj/*.o and build/mod/*.mod
make lib       # also archive the compiled satad into build/lib/libsatad.a
make driver    # compile+link the column_driver executable: build/column_driver
make clean     # remove everything under build/
```

`make check`/`make`/`make driver` compile files in a hand-ordered dependency
list inside the Makefile (Fortran modules must be compiled before anything
that `USE`s them) -- do not run with `make -j` unless you first add proper
per-object `.mod` prerequisites.

## Running

```
make run       # builds column_driver if needed, then runs it
# or, equivalently:
make driver && ./build/column_driver
```

The driver reads one CSV file and writes one back out:

- `example/fields.csv` -- one header row, then `nlev` data rows, columns:
  `rho,tk,qv,qc` (total density [kg/m^3], temperature [K], specific vapour and
  cloud-water content [kg/kg]).
- `example/output_fields.csv` -- the same columns with the post-adjustment
  state of the column.

`nlev` is inferred from the number of data rows in `fields.csv`. To stdout,
the driver prints the per-level change (`dT`, `dqv`, `dqc`) satad made, which
is the quickest way to see it working.

The bundled example is the same 10-level `rho`/`tk` profile as the parent
sandbox, with `qv`/`qc` chosen to exercise both branches of satad: dry
subsaturated levels pass through unchanged; a level with cloud water in
subsaturated air evaporates it (cooling); and moist supersaturated levels
condense vapour into cloud (warming).

The iteration controls (`maxiter`, `tol`) and the filenames are hardcoded
`PARAMETER`s near the top of `column_driver.f90` -- edit and rebuild to change
them.

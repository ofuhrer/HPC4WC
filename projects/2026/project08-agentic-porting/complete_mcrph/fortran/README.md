# hpc4wc_mcrph_column_gt4py

A single-column sandbox version of ICON's 2-moment bulk cloud microphysics
scheme (Seifert & Beheng 2006, originally `inwp_gscp == 5`). The scheme has
been decoupled from ICON's block/grid-point/domain-nesting infrastructure so
it runs on one vertical column at a time, driven by a small top-level
program (`column_driver.f90`) that reads column inputs from CSV files, calls the
scheme once, and writes the result back out.

## Layout

```
column_driver.f90            top-level program: CSV in -> scheme call -> CSV out
mo_nwp_gscp_interface.f90    Scaffolding that calls saturation adjustment (mo_satad.f90),
                             then calls the microphysics scheme, and then saturation adjustment again.
mo_2mom*.f90                 the microphysics scheme itself
dependencies/*.f90           small infrastructure the scheme needs
                             to compile (mo_kind, mo_exception, mo_physical_constants,
                             mo_lookup_tables_constants, mo_timer, mo_reff_types) - this is a mix
                             of non-physics things and files that provide physical constants 
include/*.incf               Fortran INCLUDE files pulled in by mo_2mom_mcrph_processes.f90
example/                     sample input column (hhl.csv, fields.csv) and
                             where the driver writes output_fields.csv
unused_context_dependencies/ original ICON modules kept for reference only --
                             not part of the build
Makefile                     build rules (see below)
```

## Dependencies

You need a Fortran compiler and the netCDF-Fortran library (used by the
scheme's wet-growth lookup-table file I/O). On macOS with
[Homebrew](https://brew.sh):

```
brew install gcc              # provides gfortran
brew install netcdf-fortran   # provides nf-config; pulls in netcdf, hdf5, etc.
```

This gives you `gfortran-<version>` (e.g. `gfortran-16`) and `nf-config`/
`nc-config` on your `PATH`. The Makefile auto-detects both via `nf-config`/
`nc-config`, so no further configuration is needed once they're installed.
On Linux, install your distribution's `gfortran` and `libnetcdff-dev` (or
equivalent) packages instead; anything providing `nf-config` on `PATH` will
be picked up the same way.


## Building

The default `FC` in the Makefile is `gfortran`. If `gfortran` isn't on your 
`PATH` for some reason, pass `FC=` explicitly, e.g. `make check FC=gfortran-16`.

```
make check     # fast syntax/interface check only, no code generation
make           # full compile to build/obj/*.o and build/mod/*.mod
make lib       # also archive the compiled scheme into build/lib/libtwomom.a
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

The driver reads two CSV files and writes one back out:

- `example/hhl.csv` -- one column `hhl`: `nlev+1` half-level heights [m],
  index 1 = model top, index `nlev+1` = surface (height decreases with
  index, matching ICON's convention).
- `example/fields.csv` -- one header row, then `nlev` data rows, columns:
  `rho,pres,w,tk,qv,ssat,qc,qnc,qr,qnr,qi,qni,qs,qns,qg,qng,qh,qnh,nccn,ninpot,ninagi,ninact,qrsflux`.
- `example/output_fields.csv` -- the same columns, with the post-call state
  of the column. Surface precipitation rates (rain/ice/snow/graupel/hail and
  their sum) aren't column fields, so they're printed to stdout instead.

`nlev` is inferred from the number of data rows in `fields.csv` (cross-checked
against `hhl.csv`, which must have exactly one more row). Layer thickness
`dz` is not read in -- it's computed as `dz(k) = hhl(k) - hhl(k+1)`. Note
that the `ssat` column is read but currently inert: saturation state is
driven purely by `qv` vs. the saturation specific humidity at that level's
`tk`/`rho` (via `satad`), so to set up a supersaturated layer, raise `qv`
directly rather than `ssat`.

The physics switches (timestep, saturation adjustment on/off, ice
nucleation choice, etc.) are hardcoded as `PARAMETER`s near the top of
`column_driver.f90` -- edit them there and rebuild to change them.

Filenames are hardcoded `PARAMETER`s in `column_driver.f90` too
(`hhl_file`, `fields_file`, `output_file`); edit and rebuild to point at
different data.

### First-run side effects

On its first run (per machine/build), the scheme generates two wet-growth
lookup tables (for graupel and hail) and caches them as NetCDF files
(`dmin_wetgrowth_lookup_*.nc`) in the working directory, since none exist
yet to read. This is a one-time cost -- later runs will find and reuse
them. These `.nc` files (along with `build/`, `logs/`, and
`example/output_fields.csv`) are gitignored.


## Development guidelines

When your agent has made a plan and provides a markdown file for it, please
copy that file into the `claude_plans` directory with a useful filename so
that we have some traceability of what our agents have been doing.

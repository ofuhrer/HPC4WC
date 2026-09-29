# minimal_mcrph / fortran

A single-column sandbox for a **stripped-down** version of ICON's 2-moment bulk
microphysics (Seifert & Beheng 2006) -- an intermediate step between the
saturation-adjustment-only sandbox in [`../../satad_only/fortran`](../../satad_only/fortran)
and the full scheme in [`../../fortran`](../../fortran).

It keeps only the microphysical processes that need **neither incomplete-gamma
function lookup tables nor a vertical (sedimentation) solver**, which makes it a
much smaller and more tractable target for the GT4Py port.

## What it does

Per call, on one vertical column (`nwp_microphysics` in
`mo_nwp_gscp_interface.f90`):

1. saturation adjustment (`satad_v_3d`)
2. a **reduced** `clouds_twomoment`:
   1. CCN activation (`ccn_activation_hdcp2` / `ccn_activation_sk_4d`)
   2. INP nucleation (`ice_nucleation_homhet`)
   3. cloud-droplet freezing (`cloud_freeze`)
   4. ice depositional growth (`vapor_dep_relaxation`)
   5. ice-crystal melting (`ice_melting`)
3. saturation adjustment again (`satad_v_3d`)

**Everything else in the full scheme is removed**: all ice/snow/graupel/hail
collision, riming, wet-growth conversion and rain-freezing steps; the
precipitation-size melting/evaporation steps; warm-rain
autoconversion/accretion and rain evaporation; `cloud_mass_growth`; and **all
sedimentation** (both the explicit and the semi-implicit solver). With no
precipitation-size particles there is no surface precipitation and no vertical
coupling -- every retained process acts level-by-level.

## How it differs from the full sandbox

- `mo_2mom_mcrph_processes.f90` contains **only** the ~24 routines in the
  retained call tree (7290 → ~2118 lines).
- `mo_2mom_mcrph_util.f90` has had the incomplete-gamma lookup machinery
  (`gamlookuptable`, the `incgfct_*` family) **and** the wet-growth `dmin`
  lookup / netCDF code removed (2033 → ~520 lines).
- `mo_2mom_mcrph_dmin_wetgrowth.f90` is gone, and with it the whole scheme's
  only netCDF dependency -- **this build needs no netCDF** (like `satad_only`).
- `mo_2mom_mcrph_main.f90` (`clouds_twomoment`, `init_2mom_scheme_once`) and
  `mo_2mom_mcrph_driver.f90` (`two_moment_mcrph`, `two_moment_mcrph_init`) are
  trimmed to the retained processes and setup; the driver always runs the
  explicit path (`cfg_params%i2mom_solver` is ignored).
- `include/` keeps only `phillips_nucleation_2010.incf` (feeds the retained
  Phillips/INAS ice nucleation); the LWF-melting `hailcoeffs.incf` /
  `grplcoeffs.incf` are removed.

The retained physics is otherwise **bit-identical** to the full scheme.

## Layout

```
column_driver.f90            top-level program: CSV in -> scheme call -> CSV out
mo_nwp_gscp_interface.f90    satad -> reduced scheme -> satad
mo_2mom_mcrph_driver.f90     column driver for the scheme (no sedimentation)
mo_2mom_mcrph_main.f90       reduced clouds_twomoment + trimmed one-time init
mo_2mom_mcrph_processes.f90  only the retained process routines
mo_2mom_mcrph_util.f90       special functions (no gamma-lookup / netCDF code)
mo_2mom_mcrph_{types,config,config_default}.f90, mo_2mom_prepare.f90, mo_satad.f90
mo_stage_dump.f90            opt-in per-boundary state dumps (output only; a
                             no-op unless the driver is given a dump directory)
dependencies/*.f90           mo_kind, mo_exception, mo_physical_constants,
                             mo_lookup_tables_constants, mo_timer, mo_reff_types
include/phillips_nucleation_2010.incf   INAS/Phillips nucleation lookup data
example/                     sample input column (hhl.csv, fields.csv) and where
                             the driver writes output_fields.csv
agents_plan/PLAN.md          the porting plan for this sandbox
Makefile                     netCDF-free build rules
```

## Dependencies

Just a Fortran compiler -- **no netCDF**. On macOS with
[Homebrew](https://brew.sh): `brew install gcc` (provides `gfortran`). On Linux,
install your distribution's `gfortran` package.

## Building & running

```
make check     # fast syntax/interface check only, no code generation
make           # full compile to build/obj/*.o and build/mod/*.mod
make driver    # compile+link the column_driver executable
make run       # build if needed, then run on the example column
make clean     # remove everything under build/
```

The build order is a hand-sorted list in the Makefile; do not run with `make -j`
without adding per-object `.mod` prerequisites.

The driver reads `example/hhl.csv` and `example/fields.csv` and writes
`example/output_fields.csv` (same columns, post-call state). Surface
precipitation rates are printed to stdout and are always zero here (no
sedimentation). Physics switches are hardcoded `PARAMETER`s near the top of
`column_driver.f90` -- note `ice_type = 1`, which selects `nuc_i_typ = 1` and so
turns on ice nucleation (see below).

`output_fields.csv` is a build product, not a checked-in file: it is matched by
`**/output_fields.csv` in the repo-root `.gitignore`, so a fresh clone has none.
Produce it with `make run`, which builds the driver, runs it on the bundled
column, and tees the output to `logs/driver_run.log`:

```bash
cd minimal_mcrph/fortran
make run
```

Regenerate it after changing anything here, or it describes the previous version
of the scheme. The GT4Py test suite does *not* read it -- it builds and runs this
driver itself -- so regenerating is only needed when you want to read the column
by hand, as the worked example below does.

### Command-line arguments

Paths and the timestep are defaults, not fixed: each can be overridden
positionally, and any argument left off keeps its default, so a bare
`./build/column_driver` (what `make run` does) behaves exactly as before.

```
./build/column_driver [fields.csv] [output.csv] [hhl.csv] [dt] [stage-dump-dir]
```

The fifth argument turns on `mo_stage_dump.f90`, which writes the column state
at each process boundary (`satad_pre`, `prepare`, `ccn`, `default_n`, `ice_nuc`,
`cloud_freeze`, `vapor_dep`, `ice_melt`, `post`) plus the one-time coefficient
setup (`coeffs.csv`), all at 17 significant digits. Omit it and every dump entry
point returns immediately. This is what lets the GT4Py harness compare the two
implementations process by process rather than only at the end of the timestep.

```bash
./build/column_driver ../example/fields.csv /tmp/out.csv ../example/hhl.csv 30.0 /tmp/stages
```

### The bundled example column

`example/` is a deep single column (surface → ~11 km, ISA-like) chosen to
exercise the whole retained ice/mixed-phase call tree in one shot:

- **cold cloud layers** (T ≲ −30 °C, top): liquid-supersaturated with seeded
  cloud water and a little ice → the pre-microphysics `satad` makes cloud water,
  `cloud_freeze` freezes it to ice, and `vapor_dep_relaxation` grows the ice;
- **mixed-phase deposition layers** (−30 … 0 °C): ice-supersaturated but
  liquid-subsaturated, with seeded ice → `vapor_dep_relaxation` grows ice by the
  Wegener–Bergeron–Findeisen mechanism while `ice_nucleation_homhet` activates
  INP (`ninact` rises from 0);
- **warm layers** (T > 0 °C, bottom): seeded ice → `ice_melting` converts it to
  rain.

Running it, you should see (in `output_fields.csv`): the top layers lose `qc`
and gain `qi`; the mid layers grow `qi` and drop `ninact`→activated; the bottom
layers lose `qi` and gain `qr`; temperatures rise where latent heat is released.

Because there is no wet-growth lookup table, this sandbox produces **no**
`dmin_wetgrowth_lookup_*.nc` files (unlike the full sandbox).

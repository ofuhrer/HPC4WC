# Makefile for the single-column ICON 2-moment bulk microphysics sandbox.
#
# Layout:
#   ./*.f90            the microphysics scheme itself (converted to single-column)
#   dependencies/*.f90 small non-physics ICON infrastructure modules the scheme
#                       needs to compile (mo_kind, mo_exception, mo_physical_constants, ...)
#   include/*.incf      Fortran INCLUDE files pulled in by mo_2mom_mcrph_processes.f90
#
# column_driver.f90 is the one PROGRAM unit: a top-level script that reads a
# 1D column from example/*.csv, calls the scheme once, and writes the result
# back out. Everything else here builds the scheme itself to object files +
# .mod files (and can be archived into a static library).

FC      := gfortran
INCDIR  := include
DEPDIR  := dependencies
BUILDDIR:= build
MODDIR  := $(BUILDDIR)/mod
OBJDIR  := $(BUILDDIR)/obj
LIBDIR  := $(BUILDDIR)/lib
LIBNAME := $(LIBDIR)/libtwomom.a
LOGDIR  := logs
RUN_LOG := $(LOGDIR)/driver_run.log

FFLAGS  := -cpp -ffree-line-length-none -I$(INCDIR) -J$(MODDIR)

# netCDF-Fortran is needed by mo_2mom_mcrph_util.f90 / mo_2mom_mcrph_dmin_wetgrowth.f90
# (lookup-table file I/O). Auto-detected via nf-config/nc-config if on PATH;
# both are needed since nf-config alone doesn't point at libnetcdf's -L dir.
NF_CONFIG := $(shell command -v nf-config 2>/dev/null)
NC_CONFIG := $(shell command -v nc-config 2>/dev/null)
ifneq ($(NF_CONFIG),)
  FFLAGS += $(shell nf-config --fflags)
  NETCDF_LIBS := $(shell nf-config --flibs)
  ifneq ($(NC_CONFIG),)
    NETCDF_LIBS += -L$(shell nc-config --prefix)/lib
  endif
else
  NETCDF_LIBS :=
endif

# Build order matters: Fortran modules must be compiled before anything that
# USEs them. This list is topologically sorted by hand -- do not reorder
# without checking each file's USE statements. Do not build with `make -j`
# unless you add proper per-object .mod prerequisites; the default serial
# make (no -j) walks this list in order, which is what makes it work.
SRCS := \
  $(DEPDIR)/mo_kind.f90 \
  $(DEPDIR)/mo_physical_constants.f90 \
  $(DEPDIR)/mo_lookup_tables_constants.f90 \
  $(DEPDIR)/mo_exception.f90 \
  $(DEPDIR)/mo_timer.f90 \
  $(DEPDIR)/mo_reff_types.f90 \
  mo_2mom_mcrph_config.f90 \
  mo_2mom_mcrph_config_default.f90 \
  mo_2mom_mcrph_types.f90 \
  mo_2mom_mcrph_dmin_wetgrowth.f90 \
  mo_2mom_mcrph_util.f90 \
  mo_satad.f90 \
  mo_2mom_mcrph_processes.f90 \
  mo_2mom_mcrph_main.f90 \
  mo_2mom_prepare.f90 \
  mo_2mom_mcrph_driver.f90 \
  mo_nwp_gscp_interface.f90

OBJS := $(addprefix $(OBJDIR)/,$(notdir $(SRCS:.f90=.o)))

DRIVER_SRC := column_driver.f90
DRIVER_EXE := $(BUILDDIR)/column_driver

vpath %.f90 $(DEPDIR) .

.NOTPARALLEL:
.PHONY: all check lib driver run clean

all: $(OBJS)

# Fast syntax/interface check only (no code generation) -- catches signature
# and type-shape mismatches without needing netCDF linked, much quicker than
# a full build.
check: FFLAGS += -fsyntax-only
check: $(OBJS)

lib: $(LIBNAME)

$(LIBNAME): $(OBJS) | $(LIBDIR)
	ar rcs $@ $(OBJS)

# Compiles+links the driver program directly against the already-built
# scheme objects (relies on $(OBJS) being built first for its .mod files).
driver: $(DRIVER_EXE)

$(DRIVER_EXE): $(OBJS) $(DRIVER_SRC)
	$(FC) $(FFLAGS) $(OBJS) $(DRIVER_SRC) -o $@ $(NETCDF_LIBS)

# Build and run the driver on the example column in one go. Output is both
# shown on the terminal and saved to $(RUN_LOG) (via `tee`, so the driver's
# own exit status -- not tee's -- is what determines success/failure).
run: driver | $(LOGDIR)
	./$(DRIVER_EXE) 2>&1 | tee $(RUN_LOG)

$(OBJDIR)/%.o: %.f90 | $(OBJDIR) $(MODDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR) $(MODDIR) $(LIBDIR) $(LOGDIR):
	mkdir -p $@

clean:
	rm -rf $(BUILDDIR)

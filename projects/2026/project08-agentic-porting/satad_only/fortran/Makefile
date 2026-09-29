# Makefile for the single-column ICON saturation-adjustment sandbox.
#
# This is the "satad only" cut-down of the full 2-moment microphysics sandbox
# in ../../fortran: it builds and runs mo_satad.f90 (saturation adjustment) on
# its own, with none of the rest of the scheme. Saturation adjustment acts on
# each level independently and needs no lookup-table file I/O, so unlike the
# full sandbox there is no netCDF dependency and no include/ files.
#
# Layout:
#   ./mo_satad.f90     the saturation-adjustment routine itself
#   dependencies/*.f90 the handful of ICON infrastructure / physical-constant
#                       modules it needs to compile (mo_kind,
#                       mo_physical_constants, mo_lookup_tables_constants)
#
# column_driver.f90 is the one PROGRAM unit: a top-level script that reads a
# 1D column from example/fields.csv, calls satad once, and writes the result
# back out. Everything else here builds satad itself to object + .mod files
# (and can be archived into a static library).

FC      := gfortran
DEPDIR  := dependencies
BUILDDIR:= build
MODDIR  := $(BUILDDIR)/mod
OBJDIR  := $(BUILDDIR)/obj
LIBDIR  := $(BUILDDIR)/lib
LIBNAME := $(LIBDIR)/libsatad.a
LOGDIR  := logs
RUN_LOG := $(LOGDIR)/driver_run.log

FFLAGS  := -cpp -ffree-line-length-none -J$(MODDIR)

# Build order matters: Fortran modules must be compiled before anything that
# USEs them. This list is topologically sorted by hand -- do not reorder
# without checking each file's USE statements. Do not build with `make -j`
# unless you add proper per-object .mod prerequisites; the default serial
# make (no -j) walks this list in order, which is what makes it work.
SRCS := \
  $(DEPDIR)/mo_kind.f90 \
  $(DEPDIR)/mo_physical_constants.f90 \
  $(DEPDIR)/mo_lookup_tables_constants.f90 \
  mo_satad.f90

OBJS := $(addprefix $(OBJDIR)/,$(notdir $(SRCS:.f90=.o)))

DRIVER_SRC := column_driver.f90
DRIVER_EXE := $(BUILDDIR)/column_driver

vpath %.f90 $(DEPDIR) .

.NOTPARALLEL:
.PHONY: all check lib driver run clean

all: $(OBJS)

# Fast syntax/interface check only (no code generation) -- catches signature
# and type-shape mismatches, much quicker than a full build.
check: FFLAGS += -fsyntax-only
check: $(OBJS)

lib: $(LIBNAME)

$(LIBNAME): $(OBJS) | $(LIBDIR)
	ar rcs $@ $(OBJS)

# Compiles+links the driver program directly against the already-built
# satad objects (relies on $(OBJS) being built first for its .mod files).
driver: $(DRIVER_EXE)

$(DRIVER_EXE): $(OBJS) $(DRIVER_SRC)
	$(FC) $(FFLAGS) $(OBJS) $(DRIVER_SRC) -o $@

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

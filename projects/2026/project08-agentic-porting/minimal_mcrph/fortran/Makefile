# Makefile for the single-column ICON "minimal" two-moment microphysics sandbox.
#
# This is the cut-down sibling of the full 2-moment sandbox in ../../fortran and
# the satad-only sandbox in ../../satad_only/fortran. It runs saturation
# adjustment plus a *reduced* microphysics call tree (CCN activation, INP
# nucleation, cloud-droplet freezing, ice depositional growth, ice melting) and
# nothing else -- no collisions, riming, wet-growth conversion, rain processes,
# or sedimentation.
#
# Because the retained processes need neither the incomplete-gamma lookup tables
# nor the wet-growth dmin lookup table, this build has **no netCDF dependency**
# (unlike ../../fortran): mo_2mom_mcrph_dmin_wetgrowth.f90 is gone and the
# netCDF-using parts of mo_2mom_mcrph_util.f90 have been removed.
#
# Layout:
#   ./*.f90            the (reduced) microphysics scheme, single-column
#   dependencies/*.f90 small ICON infrastructure / physical-constant modules
#   include/*.incf     Fortran INCLUDE files pulled in by mo_2mom_mcrph_processes.f90
#
# column_driver.f90 is the one PROGRAM unit: it reads a 1D column from
# example/, calls the scheme once, and writes the result back out.

FC      := gfortran
INCDIR  := include
DEPDIR  := dependencies
BUILDDIR:= build
MODDIR  := $(BUILDDIR)/mod
OBJDIR  := $(BUILDDIR)/obj
LIBDIR  := $(BUILDDIR)/lib
LIBNAME := $(LIBDIR)/libtwomom_minimal.a
LOGDIR  := logs
RUN_LOG := $(LOGDIR)/driver_run.log

FFLAGS  := -cpp -ffree-line-length-none -I$(INCDIR) -J$(MODDIR)

# Build order matters: Fortran modules must be compiled before anything that
# USEs them. This list is topologically sorted by hand -- do not reorder without
# checking each file's USE statements. Do not build with `make -j` unless you
# add proper per-object .mod prerequisites; the default serial make walks this
# list in order, which is what makes it work.
SRCS := \
  $(DEPDIR)/mo_kind.f90 \
  $(DEPDIR)/mo_physical_constants.f90 \
  $(DEPDIR)/mo_lookup_tables_constants.f90 \
  $(DEPDIR)/mo_exception.f90 \
  $(DEPDIR)/mo_timer.f90 \
  $(DEPDIR)/mo_reff_types.f90 \
  mo_stage_dump.f90 \
  mo_2mom_mcrph_config.f90 \
  mo_2mom_mcrph_config_default.f90 \
  mo_2mom_mcrph_types.f90 \
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

# Fast syntax/interface check only (no code generation).
check: FFLAGS += -fsyntax-only
check: $(OBJS)

lib: $(LIBNAME)

$(LIBNAME): $(OBJS) | $(LIBDIR)
	ar rcs $@ $(OBJS)

driver: $(DRIVER_EXE)

$(DRIVER_EXE): $(OBJS) $(DRIVER_SRC)
	$(FC) $(FFLAGS) $(OBJS) $(DRIVER_SRC) -o $@

run: driver | $(LOGDIR)
	./$(DRIVER_EXE) 2>&1 | tee $(RUN_LOG)

$(OBJDIR)/%.o: %.f90 | $(OBJDIR) $(MODDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR) $(MODDIR) $(LIBDIR) $(LOGDIR):
	mkdir -p $@

clean:
	rm -rf $(BUILDDIR)

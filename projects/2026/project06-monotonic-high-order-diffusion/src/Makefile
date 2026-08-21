SHELL = /bin/sh
HOSTNAME = $(shell hostname | sed 's/[0-9]*//g' | sed 's/\..*//g')

VERSION ?= base

# Sources live in subdirectories by provenance; VPATH keeps the build and
# the binaries flat in src/.
VPATH = baseline:filter:gt4py:tests

TARGET = stencil2d-$(VERSION).x

SRCS = m_utils.F90 \
       m_partitioner.F90 \
       stencil2d-$(VERSION).F90

OBJS := $(SRCS:.F90=.o)

# CSCS Alps (Grace, GH200): GNU toolchain from the uenv, so build inside
# `uenv run prgenv-gnu/26.3:v1 --view=default -- make ...`
ifeq "$(HOSTNAME)" "santis-ln"
  CSCS_GNU = 1
endif
ifeq "$(HOSTNAME)" "nid"
  CSCS_GNU = 1
endif
ifdef CSCS_GNU
  F90 = mpif90
  FFLAGS = -cpp -DGNU_MPI_WORKAROUND -O3 -mcpu=native -funroll-loops -ffree-line-length-none
ifdef NOOPT
  FFLAGS = -cpp -DGNU_MPI_WORKAROUND -O0 -g -ffree-line-length-none
  TARGET = stencil2d-$(VERSION)-noopt.x
endif
endif

# generic fallback: gfortran via mpif90 (local development)
ifndef F90
  F90 = mpif90
  FFLAGS = -cpp -DGNU_MPI_WORKAROUND -O3 -march=native -funroll-loops -ffree-line-length-none
ifdef DEBUG
  FFLAGS = -cpp -DGNU_MPI_WORKAROUND -O0 -g -fbacktrace -fno-fast-math -ffree-line-length-none \
    -pedantic -Waliasing -Wampersand -Wline-truncation -Wsurprising -Wtabs -Wunderflow \
    -ffpe-trap=invalid,zero,overflow -fbounds-check -finit-real=nan
  TARGET = stencil2d-$(VERSION)-debug.x
endif
endif

.PHONY : clean distclean

$(TARGET): $(OBJS)
	$(F90) $(FFLAGS) $(OBJS) -o $(TARGET)

clean:
	-$(RM) -rf *.o *.mod *.x *.dSYM *.dat perf/

distclean: clean
	-$(RM) -rf *.png *.npy

%.o: %.F90
	$(F90) $(FFLAGS) -c $<

test_sg.x: m_utils.o m_partitioner.o test_scatter_gather.F90
	$(F90) $(FFLAGS) $^ -o $@

VERSIONS = base limiter filter filter-kblock filter-inline filter-zalesak

.PHONY: allversions
allversions:
	@for v in $(VERSIONS); do \
	  $(MAKE) --no-print-directory VERSION=$$v || exit 1; \
	done
	@$(MAKE) --no-print-directory test_sg.x

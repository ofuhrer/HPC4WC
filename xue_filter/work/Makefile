SHELL = /bin/sh

TARGET := main.x  # name of executable

SRCS := m_constants.F90 \
	m_cdf.F90 \
	m_compute.F90 \
	m_utils.F90 \
    main.F90

DEPF := .depend

############## EDITABLE SECTION ENDS HERE ##############

# create object file list
OBJS := $(SRCS:.F90=.o)

# setup
USE_MPI = Y
USE_IO = Y

# default flags
F90  = 
FDBG = 
FOPT = 
FINC = 
FLD  = 

# machine dependent section
OS:= $(shell uname -s)
ifeq ($(OS),Darwin)
  FNAME = gnu
ifeq ($(USE_MPI),Y)
  F90 = mpif90 -cpp
else
  F90 = gfortran -cpp
endif
endif
ifeq ($(OS),Linux)
  FNAME := gnu
ifeq ($(USE_MPI),Y)
  F90 = mpif90 -cpp
else
  F90 = gfortran -cpp
endif
endif
ifndef F90
  $(error No default compiler present for $(OS) platform!)
endif

# compiler dependent section
ifeq ($(FNAME),gnu)
  FDBG += -O0 -g -fbacktrace -fno-fast-math -ffree-line-length-none -fno-backslash \
	-pedantic -Waliasing -Wampersand -Wline-truncation -Wsurprising -Wtabs -Wunderflow \
	-fdump-core -ffpe-trap=invalid,zero,overflow -fbounds-check -finit-real=nan \
	-finit-integer=9999999 -finit-logical=true -finit-character=35
  FOPT += -O2
  FINC += -cpp -I.
  FLD  += 
endif

# choose compiler options
ifdef DEBUG
  FFLAGS += $(FDBG)
else
  FFLAGS += $(FOPT)
endif

# MPI section
ifeq ($(USE_MPI),Y)
  FINC += -DUSE_MPI
endif

# NetCDF section
ifeq ($(USE_IO),Y)
ifeq ($(OS),Darwin)
  FINC += -DUSE_IO -I/usr/local/include
  FLD  += -L/usr/local/lib -lnetcdf -lnetcdff
endif
ifeq ($(OS),Linux)
  FINC += -DUSE_IO -I/usr/include
  FLD  += -L/usr/lib -lnetcdf -lnetcdff
endif
endif

# definition of phony targets

.PHONY : opt debug test clean depend

opt: depend
	$(MAKE) $(TARGET)

debug: depend
	$(MAKE) DEBUG=debug $(TARGET)

test: 
	( cd test/; ./clean_all.sh; ./run_all.sh )

clean: 
	-$(RM) *~ *.nc $(DEPF)* *.o *.mod *.MOD

depend: 
	$(MAKE) $(DEPF)

# definition of targets and rules

$(TARGET) : $(DEPF) $(OBJS)
	$(F90) $(FFLAGS) $(FINC) $(OBJS) $(FLD) -o $(TARGET)

$(DEPF) : $(SRCS)
	-$(RM) $(DEPF)*
	./sfmakedepend --case down --file $(DEPF) $(SRCS)
	ln -s $(DEPF) $(DEPF).inc

%.o : %.f90
	$(F90) $(FFLAGS) $(FINC) -c $*.f90

%.o : %.F90
	$(F90) $(FFLAGS) $(FINC) -c $*.F90

# include dependency file

-include $(DEPF).inc

# DO NOT DELETE THIS LINE - used by make depend

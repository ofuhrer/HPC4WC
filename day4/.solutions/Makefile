SHELL = /bin/sh
HOSTNAME = $(shell hostname | sed 's/[0-9]*//g' | sed 's/\..*//g')

VERSION ?= orig

TARGET = stencil2d-$(VERSION).x

SRCS = m_utils.F90 \
       m_partitioner.F90 \
       stencil2d-$(VERSION).F90

# create object file list
OBJS := $(SRCS:.F90=.o)

ifeq "$(HOSTNAME)" "daint"
  F90 = ftn
  FFLAGS = -O3 -hfp3 -eZ -ffree -N255 -ec -eC -eI -eF -rm
ifdef NOOPT
  FFLAGS = -O0 -hfp0 -Oipa0 -Ovector0 -ffree -N255 -ec -eC -eI -eF -rm -acc
  TARGET = stencil2d-$(VERSION)-noopt.x
endif
endif
ifeq "$(HOSTNAME)" "nid"
  F90 = ftn
  FFLAGS = -O3 -hfp3 -eZ -ffree -N255 -ec -eC -eI -eF -rm
ifdef NOOPT
  FFLAGS = -O0 -hfp0 -Oipa0 -Ovector0 -ffree -N255 -ec -eC -eI -eF -rm -acc
  TARGET = stencil2d-$(VERSION)-noopt.x
endif
endif
ifeq "$(HOSTNAME)" "papaya"
  F90 = mpif90 -cpp 
  FFLAGS = -O0 -g -fbacktrace -acc -fno-fast-math -ffree-line-length-none -fno-backslash \
    -pedantic -Waliasing -Wampersand -Wline-truncation -Wsurprising -Wtabs -Wunderflow \
    -fdump-core -ffpe-trap=invalid,zero,overflow -fbounds-check -finit-real=nan \
    -finit-integer=9999999 -finit-logical=true -finit-character=35
endif
ifndef F90
  $(error F90 is not set for $(HOSTNAME))
endif

# definition of phony targets

.PHONY : clean

$(TARGET): $(OBJS)
	$(F90) $(FFLAGS) $(OBJS) -o $(TARGET)
	cp $(TARGET) stencil2d.x

clean: 
	-$(RM) -rf *~ *.o *.mod *.MOD *.i core.* *.out *.lst *.x *.x+orig *.x+[0-9]* *.dat *.report result_*.py report*.txt

distclean: clean
	-$(RM) -rf *.png *.dat *.npy *.x *.x+orig* *.x+[0-9]*

%.o: %.F90
	$(F90) $(FFLAGS) -c $*.F90

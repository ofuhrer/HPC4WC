SHELL = /bin/sh

TARGET = stencil_2d.x

SRCS = m_utils.F90 \
       stencil_2d.F90

# create object file list
OBJS := $(SRCS:.F90=.o)

F90 = ftn
FFLAGS = -eZ -ffree -N255 -ec -eC -eI -eF

# definition of phony targets

.PHONY : clean

$(TARGET): $(OBJS)
	$(F90) $(FFLAGS) $(OBJS) -o $(TARGET)

clean: 
	-$(RM) -rf *~ *.o *.mod *.MOD *.i *.x *.x+orig *.x+[0-9]* core.* *.out

%.o: %.F90
	$(F90) $(FFLAGS) -c $*.F90

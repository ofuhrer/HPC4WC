SERIALBOX_DIR ?= /usr/local/serialbox
NETCDF_LIB ?= /lib/x86_64-linux-gnu

TARGET = main.x

SRC = \
  m_countdown.F \
  machine.F \
  physparam.F \
  physcons.F \
  funcphys.F \
  sfc_sice.F \
  main.F

OBJ = $(SRC:.F=.o)
PP = $(SRC:.F=.F90)

F90 = gfortran

# non-optimized compiler options
FFLAGS = -fdec -fdefault-real-8 -fno-fast-math -ffree-form -ffree-line-length-none \
  -fno-backslash -fimplicit-none -frange-check -pedantic -Waliasing -Wampersand \
  -Wline-truncation -Wsurprising -Wtabs -Wunderflow -O0 -g -fbacktrace \
  -fdump-core -ffpe-trap=invalid,zero,overflow -fbounds-check -finit-real=nan \
  -finit-integer=9999999 -finit-logical=true -finit-character=35 \
  -DSERIALIZE -I$(SERIALBOX_DIR)/include
  
# optimzed compiler options
#FFLAGS = -fdec -fdefault-real-8 -fno-fast-math -ffree-form -ffree-line-length-none \
  -fno-backslash -fimplicit-none -pedantic -Waliasing -Wampersand \
  -Wline-truncation -Wsurprising -Wtabs -Wunderflow -O3 -ftree-vectorize -funroll-loops \
  -DSERIALIZE -I$(SERIALBOX_DIR)/include
  
LD = gfortran
LDFLAGS = $(SERIALBOX_DIR)/lib/libSerialboxFortran.a $(SERIALBOX_DIR)/lib/libSerialboxC.a \
  $(SERIALBOX_DIR)/lib/libSerialboxCore.a -L$(NETCDF_LIB) -lnetcdff -lnetcdf \
  -lpthread -lstdc++ -lstdc++fs

.PHONY: all
all: pre-process $(TARGET)

.PHONY: pre-process
pre-process: $(PP)

.PHONY: clean
clean:
	/bin/rm -rf *.mod *.o $(PP) data.tar.gz __pycache__ $(TARGET) dump/*

.PHONY: distclean
distclean: clean
	/bin/rm -rf data dump

$(TARGET): $(OBJ)
	$(LD) $(OBJ) $(LDFLAGS) -o $(TARGET)

%.F90: %.F
	$(SERIALBOX_DIR)/python/pp_ser/pp_ser.py --no-prefix -v --output=$@ $<

%.o: %.F90
	$(F90) $(FFLAGS) -c $< -o $@


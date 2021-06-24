import sys, os
from shalconv import DATAPATH, SERIALBOX_DIR, ISDOCKER
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

if "NETCDF_LIB" in os.environ:
    NETCDF_LIB = os.environ["NETCDF_LIB"]
else:
    NETCDF_LIB = "/usr/lib/x86_64-linux-gnu"

FFLAGS = f"-cpp -fdec -fdefault-real-8 -fno-fast-math -ffree-form -ffree-line-length-none \
           -fno-backslash -fimplicit-none -frange-check -pedantic -Waliasing -Wampersand \
           -Wline-truncation -Wsurprising -Wtabs -Wunderflow -O0 -g -fbacktrace -fdump-core \
           -ffpe-trap=invalid,zero,overflow -fbounds-check -finit-real=nan -finit-integer=9999999 \
           -finit-logical=true -finit-character=35 -DSERIALIZE -I{SERIALBOX_DIR}/include"

LDFLAGS = f"{SERIALBOX_DIR}/lib/libSerialboxFortran.a {SERIALBOX_DIR}/lib/libSerialboxC.a \
           {SERIALBOX_DIR}/lib/libSerialboxCore.a -L/lib/x86_64-linux-gnu -L{NETCDF_LIB} \
            -lnetcdff -lnetcdf -lpthread -lstdc++ -lstdc++fs"

datapath     = DATAPATH
outputfile   = "fortran/samfshalconv_generated.f90"
templatefile = "fortran/samfshalconv_serialize.template.f90"
inputfile    = "fortran/samfshalconv_serialize.f90"
objfile      = "fortran/samfshalconv_generated.o"
targetfile   = "fortran/samfshalconv_generated.x"

with open(templatefile,"r") as f:
    
    filestr = f.read().replace("DATAPATH",datapath)
    if ISDOCKER:
        filestr = filestr.replace(" QUOTATION ",'"')
        filestr = filestr.replace(" QUOTATION", '"')
    else:
        filestr = filestr.replace("QUOTATION", '')
    
    with open(inputfile,"w") as fw:
        fw.write(filestr)
        
os.system(f"{SERIALBOX_DIR}/python/pp_ser/pp_ser.py --no-prefix -v --output={outputfile} {inputfile}")
os.system(f"gfortran {FFLAGS} -c {outputfile} -o {objfile} -DDATPATH='{datapath}'")
os.system(f"gfortran {objfile} {LDFLAGS} -o {targetfile}")
os.system(f"./{targetfile}")
#os.system(f"f2py --f2cmap fortran/.f2py_f2cmap -c -m samfshalconv_ser {outputfile} --f90flags='{FFLAGS}' {F2PYFLAGS}")

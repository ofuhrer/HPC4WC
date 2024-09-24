# -*- coding: utf-8 -*-
"""
File defining the global variables used in the main program
and all subfunctions.
"""

# --------------------------------------------------------
# --------------------- USER NAMELIST --------------------
# --------------------------------------------------------

# Output control
# -------------------------------------------------
out_fname = "output_optimized"  # file name of output
iout = 360  # write every iout-th time-step into the output file
iiniout = 1  # write initial field (0 = no, 1 = yes)

# Domain size
# -------------------------------------------------
xl = 5_000_000.0  # domain size  [m]
nx = 5040  # number of grid points in horizontal direction
dx = xl / nx  # horizontal resolution [m]
thl = 350.0  # domain depth  [K]
nz = 60  # vertical resolution
dt = 2  # time step [s]
diff = 0.05  # (horizontal) diffusion coefficient
time = 2 * 60 * 60  # integration time [s]

# Topography
# -------------------------------------------------
topomx = 500  # mountain height [m]
topowd = 400000  # mountain half width [m]
topotim = 1800  # mountain growth time [s]

# Initial atmosphere
# -------------------------------------------------
u00 = 22.5  # initial velocity [m/s]
bv00 = 0.015  # Brunt-Vaisalla frequency [1/s]
th00 = 300.0  # potential temperature at surface

ishear = 0  # wind shear simulation (0 = no shear, 1 = shear)
k_shl = 5  # bottom level of wind shear layer (ishear = 1)
# bottom level of wind layer is 0 (index)
k_sht = 8  # top level of wind shear layer (ishear = 1)
# top level of wind layer is nz-1 (index)
u00_sh = 10.0  # initial velocity below shear layer [m/s] (ishear = 1)
# u00 is speed above shear layer [m/s]   #orig 0.

# Boundaries
# -------------------------------------------------
nab = 30  # number of grid points in absorber
diffabs = 1.0  # maximum value of absorber
irelax = 0  # lateral boundaries (0 = periodic, 1 = relax)

nb = 2  # number of boundary points on each side

# Print options
# -------------------------------------------------
idbg = 0  # print debugging text (0 = not print, 1 = print)
iprtcfl = 0  # print Courant number (0 = not print, 1 = print)
itime = 0  # print computation time (0 = not print, 1 = print)

# Physics: Moisture
# -------------------------------------------------
imoist = 1  # include moisture (0 = dry, 1 = moist)
imoist_diff = 1  # apply diffusion to qv, qc, qr (0 = off, 1 = on)
imicrophys = 1  # include microphysics (0 = off, 1 = kessler, 2 = two moment)
idthdt = 0  # couple physics to dynamics (0 = off, 1 = on)
iern = 1  # evaporation of rain droplets (0 = off, 1 = on)

# Options for Kessler scheme
# -------------------------------------------------
vt_mult = 1.0  # multiplication factor for terminal fall velocity
autoconv_th = 0.0001  # critical cloud water mixing ratio for the onset
# of autoconversion [kg/kg]
autoconv_mult = 1.0  # multiplication factor for autoconversion
sediment_on = 1  # switch to turn on / off sedimentation

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

# Physical constants
# --------------------------
g = 9.81  # gravity
cp = 1004.0  # specific heat of air at constant pressure
r = 287.0  # gas constant of air [J/kgK]
r_v = 461.0  # gas constant of vapor [J/kgK]
rdcp = r / cp  # short cut for R/Cp
cpdr = cp / r  # short cut for Cp/R
pref = 100 * 1000.0  # reference pressure in SI units (Pa, not hPa!)
z00 = 0.0  # surface height
prs00 = pref  # upstream surface pressure (= ref. pressure)
exn00 = cp * (prs00 / pref) ** rdcp  #

# compute input parameters
# --------------------------
dth = thl / nz  # spacing between vertical layers [K]
nts = round(time / dt, 0)  # number of iterations
nout = int(nts / iout)  # number of output steps

nx1 = nx + 1  # number of staggered gridpoints in x
nz1 = nz + 1  # number of staggered gridpoints in z
nxb = nx + 2 * nb  # x range of unstaggered variable
nxb1 = nx1 + 2 * nb  # x range of staggered variable

# END OF NAMELIST.PY

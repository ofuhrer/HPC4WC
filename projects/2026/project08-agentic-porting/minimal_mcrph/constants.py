"""Physical and scheme constants used by the retained process routines.

Transcribed from the ``USE mo_physical_constants`` alias block in
``mo_2mom_mcrph_processes.f90`` (lines 89-105) and the module-level ``PARAMETER``
declarations further down the same file, not renamed or reinterpreted.

R_D_VAPOR is the one to watch: the Fortran aliases it as ``R_d`` (line 91,
``R_d => rv``), which reads like "gas constant of dry air" but is actually the gas
constant for water VAPOR (461.51). The real dry-air constant is aliased ``R_l``
(line 90, ``R_l => rd``, "luft" = air). Every ``qv * R_d * T`` expression in the
Fortran (e.g. ssi/qvsidiff calculations) uses the vapor one. Getting this backwards
silently produces a ~1.6x error in every supersaturation calculation.
"""

import math

# -- from mo_physical_constants.f90 (dependencies/), aliased as in processes.f90 --
R_DRY_AIR = 287.04  # rd, "R_l" in the Fortran alias
R_D_VAPOR = 461.51  # rv, aliased "R_d" in the Fortran -- see module docstring
CPD = 1004.64  # specific heat of dry air at constant pressure
T_MELT = 273.15  # tmelt, melting temperature of ice/snow
RHO_W = 1000.0  # rhoh2o, density of liquid water
RHO_ICE = 916.7  # rhoice, density of pure ice
NU_L = 1.50e-5  # con_m, kinematic viscosity of dry air [m^2/s]
K_T = 2.40e-2  # con0_h, thermal conductivity of dry air [J/m/s/K]
N_AVO = 6.02214179e23  # avo, Avogadro constant [1/mol]
K_B = 1.3806504e-23  # ak, Boltzmann constant [J/K]
GRAV = 9.80665  # av. gravitational acceleration [m/s2]
L_VAPORIZATION = 2.5008e6  # alv, latent heat of vaporization [J/kg]
L_SUBLIMATION = 2.8345e6  # als, latent heat of sublimation [J/kg]
L_FUSION = L_SUBLIMATION - L_VAPORIZATION  # alf = als - alv

# -- module-level PARAMETERs in mo_2mom_mcrph_processes.f90 --
PI = math.pi  # matches the Fortran's own 3.14159265358979323846 to float64 precision
N_SC = 0.710  # Schmidt number (PK, S.541), line 157
N_F = 0.333  # exponent of N_sc in ventilation coefficient (PK, S.541), line 158
NI_HET_MAX = 100.0e3  # max heterogeneous IN number density [1/m3], line 182
NI_HOM_MAX = 5000.0e3  # max homogeneous-nucleation ice number density [1/m3], line 183

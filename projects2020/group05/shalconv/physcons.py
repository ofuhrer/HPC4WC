import numpy as np


### Math constants ###
con_pi       = np.pi          # Pi
con_e        = np.e           # e
con_sqrt2    = np.sqrt(2)     # Square root of 2
con_sqrt3    = np.sqrt(3)     # Square root of 3


### Geophysics/Astronomy constants ###
con_rerth    = 6.3712e+6      # Radius of Earth
con_g        = 9.80665        # Gravity acceleration
con_omega    = 7.2921e-5      # Angular velocity of Earth
con_p0       = 1.01325e+5     # Standard atmosphere pressure
con_solr_old = 1.3660e+3      # Solar constant old
con_solr     = 1.3608e+3      # Solar constant


### Thermodynamics constants ###
con_rgas     = 8.314472       # Molar gas constant
con_rd       = 2.8705e+2      # Gas constant air
con_rv       = 4.6150e+2      # Gas constant H2O
con_cp       = 1.0046e+3      # Specific heat of air at p
con_cv       = 7.1760e+2      # Specific heat of air at v
con_cvap     = 1.8460e+3      # Specific heat of H2O gas
con_cliq     = 4.1855e+3      # Specific heat of H2O liquid
con_csol     = 2.1060e+3      # Specific heat of H2O ice
con_hvap     = 2.5000e+6      # Latent heat of H2O condensation
con_hfus     = 3.3358e+5      # Latent heat of H2O fusion
con_psat     = 6.1078e+2      # Pressure at H2O 3pt
con_t0c      = 2.7315e+2      # Temperature at 0C in Kelvin
con_ttp      = 2.7316e+2      # Temperature at H2O 3pt
con_tice     = 2.7120e+2      # Temperature of freezing sea
con_jcal     = 4.1855         # Joules per calorie
con_rhw0     = 1022.0         # Sea water reference density
con_epsq     = 1.0e-12        # Min q for computing precipitation type


### Secondary constants ###
con_rocp     = con_rd/con_cp
con_cpor     = con_cp/con_rd
con_rog      = con_rd/con_g
con_fvirt    = con_rv/con_rd - 1.
con_eps      = con_rd/con_rv
con_epsm1    = con_rd/con_rv - 1.
con_dldtl    = con_cvap - con_cliq
con_dldti    = con_cvap - con_csol
con_xponal   = -con_dldtl/con_rv
con_xponbl   = -con_dldtl/con_rv + con_hvap/(con_rv * con_ttp)
con_xponai   = -con_dldti/con_rv
con_xponbi   = -con_dldti/con_rv + (con_hvap + con_hfus)/(con_rv * con_ttp)


### Physics/Chemistry constants ###
con_c        = 2.99792458e+8  # Speed of light
con_plnk     = 6.6260693e-34  # Planck constant
con_boltz    = 1.3806505e-23  # Boltzmann constant
con_sbc      = 5.670400e-8    # Stefan-Boltzmann constant
con_avgd     = 6.0221415e+23  # Avogadro constant
con_gasv     = 22413.996e-6   # Vol. of ideal gas at 273.15K, 101.325kPa
con_amd      = 28.9644        # Molecular weight of dry air
con_amw      = 18.0154        # Molecular weight of water vapor
con_amo3     = 47.9982        # Molecular weight of O3
con_amco2    = 44.011         # Molecular weight of CO2
con_amo2     = 31.9999        # Molecular weight of O2
con_amch4    = 16.043         # Molecular weight of CH4
con_amn2o    = 44.013         # Molecular weight of N2O
con_thgni    = -38.15         # Temperature the H.G.Nuc. ice starts
cimin        = 0.15           # Minimum ice concentration
qamin        = 1.0e-16        # Minimum aerosol concentration


### Miscellaneous physics related constants
rlapse       = 0.65e-2
cb2mb        = 10.0
pa2mb        = 0.01
rhowater     = 1000.          # Density of water in kg/m³
rhosnow      = 100.           # Density of snow in kg/m³
rhoair       = 1.28           # Density of air near the surface in kg/m³
PQ0          = 379.90516
A2A          = 17.2693882
A3           = 273.16
A4           = 35.86
RHmin        = 1.0e-6

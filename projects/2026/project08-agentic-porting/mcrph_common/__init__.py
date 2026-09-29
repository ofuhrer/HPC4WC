"""Harness plumbing shared by the microphysics variants.

Nothing physical lives here. These are the parts every variant needs in the same
form -- reading and writing the column CSVs that the Fortran drivers speak,
building and running those drivers, and comparing a port's output against the
Fortran reference.
"""

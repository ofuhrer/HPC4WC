"""Saturation adjustment (ICON's ``mo_satad.f90``) in isolation: the Fortran
reference, the GT4Py port, and the harness that holds them to each other.

``FORTRAN_DIR``/``EXAMPLE_DIR`` are anchored to this file rather than to the
working directory, so tests and scripts find the reference code and the sample
column no matter where they are invoked from. Every variant exposes the same two.
"""

from pathlib import Path

VARIANT_DIR = Path(__file__).resolve().parent
FORTRAN_DIR = VARIANT_DIR / "fortran"
EXAMPLE_DIR = VARIANT_DIR / "example"

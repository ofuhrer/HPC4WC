"""The stripped two-moment scheme: activation, nucleation and depositional growth,
with precipitation and riming removed. Fortran reference plus GT4Py port.
"""

from pathlib import Path

VARIANT_DIR = Path(__file__).resolve().parent
FORTRAN_DIR = VARIANT_DIR / "fortran"
EXAMPLE_DIR = VARIANT_DIR / "example"

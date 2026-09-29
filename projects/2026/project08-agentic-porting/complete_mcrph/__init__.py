"""The full Seifert and Beheng (2006) two-moment scheme as implemented in ICON.

Fortran reference only so far -- no GT4Py port yet.
"""

from pathlib import Path

VARIANT_DIR = Path(__file__).resolve().parent
FORTRAN_DIR = VARIANT_DIR / "fortran"
EXAMPLE_DIR = VARIANT_DIR / "example"

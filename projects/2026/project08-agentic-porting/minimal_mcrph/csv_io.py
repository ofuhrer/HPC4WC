"""Read/write the same CSV format as column_driver.f90.

Column order: rho,pres,w,tk,qv,ssat,qc,qnc,qr,qnr,qi,qni,qs,qns,qg,qng,qh,qnh,
nccn,ninpot,ninagi,ninact,qrsflux (column_driver.f90:37-39). `ssat` and
`qrsflux` pass through unchanged: `ssat` is only touched when
`lexpl_supersat=True` (False for this config), `qrsflux` is only zeroed when
`ldass_lhn=True` (hardcoded False, mo_2mom_mcrph_driver.f90:89) -- neither
guard is ever entered, so neither is ever read or written by the reference
Fortran either.

`hhl.csv` is a separate one-column file with nlev+1 half-level heights.
"""

import numpy as np

from mcrph_common.csv_io import read_column, read_columns, write_columns
from minimal_mcrph.driver import ColumnState

# The driver's column order (column_driver.f90:37-39). Public because the test harness
# builds input columns against it -- anything writing a CSV this driver will read has to
# agree on both the names and their order.
COLUMNS = (
    "rho", "pres", "w", "tk", "qv", "ssat", "qc", "qnc", "qr", "qnr", "qi", "qni",
    "qs", "qns", "qg", "qng", "qh", "qnh", "nccn", "ninpot", "ninagi", "ninact", "qrsflux",
)  # fmt: skip

# ssat and qrsflux ride along in the CSV but are not part of ColumnState.
_STATE_COLUMNS = tuple(name for name in COLUMNS if name not in ("ssat", "qrsflux"))


def read_hhl_csv(path: str) -> np.ndarray:
    return read_column(path, "hhl")


def read_fields_csv(path: str, hhl_path: str) -> tuple[ColumnState, np.ndarray, np.ndarray]:
    """Returns (state, ssat, qrsflux) -- the latter two pass through unchanged."""
    cols = read_columns(path)
    hhl = read_hhl_csv(hhl_path)
    nlev = len(cols["rho"])
    if hhl.shape[0] != nlev + 1:
        raise ValueError(f"hhl.csv has {hhl.shape[0]} rows, expected nlev+1={nlev + 1}")
    state = ColumnState(hhl=hhl, **{name: cols[name] for name in _STATE_COLUMNS})
    return state, cols["ssat"], cols["qrsflux"]


def write_fields_csv(path: str, state: ColumnState, ssat: np.ndarray, qrsflux: np.ndarray) -> None:
    data = {name: getattr(state, name) for name in _STATE_COLUMNS}
    data["ssat"] = ssat
    data["qrsflux"] = qrsflux
    write_columns(path, {name: data[name] for name in COLUMNS})
